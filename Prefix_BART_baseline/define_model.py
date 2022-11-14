import torch
from transformers import BartTokenizer
from prompt_model.modeling_bart import BartForConditionalGeneration
from pytorch_lightning import LightningModule


class PrefixBartModel(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model_path = kwargs['model_path']
        self.tokenizer_path = kwargs['tokenizer_path']
        self.lr = kwargs['lr']
        self.num_beams = kwargs['num_beams']

        self.tokenizer = BartTokenizer.from_pretrained(self.tokenizer_path)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_path)
        # self.model.freeze()
        for p in self.model.parameters():
            p.requires_grad = False

        # prefix-related part
        self.best_valid_loss = float('inf')

        self.enc_mlp = torch.nn.Linear(kwargs['hidden_dim'], kwargs['hidden_dim'])
        self.dec_mlp = torch.nn.Linear(kwargs['hidden_dim'], kwargs['hidden_dim'])
        self.cross_mlp = torch.nn.Linear(kwargs['hidden_dim'], kwargs['hidden_dim'])

        self.prompt = torch.load(kwargs['prompt_file'], map_location='cuda:4')
        self.prompt = torch.load(kwargs['prompt_file'])
        assert self.prompt[0]['prev_key'].requires_grad is True

        self.off_the_shelf_prefix_path = kwargs['off_the_shelf_prefix_path']
        self.off_the_shelf_prefix = None

    def training_step(self, batch, batch_idx):
        # 确实是每个step更新一次
        # self.prompt --(mlp)--> inter_prompt --(reformat)--> past_prompt
        bsz = len(batch['input_ids'])
        inter_prompt = [
            self.enc_mlp(self.prompt[0]['prev_key'].to(self.device)),
            self.enc_mlp(self.prompt[0]['prev_value'].to(self.device)),
            self.dec_mlp(self.prompt[1]['prev_key'].to(self.device)),
            self.dec_mlp(self.prompt[1]['prev_value'].to(self.device)),
            self.cross_mlp(self.prompt[2]['prev_key'].to(self.device)),
            self.cross_mlp(self.prompt[2]['prev_value'].to(self.device))
        ]
        past_prompt = self.reformat(inter_prompt, bsz)

        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            decoder_input_ids=batch['decoder_input_ids'],
            labels=batch['labels'],
            past_prompt=past_prompt
        )
        train_loss = outputs[0]
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        bsz = len(batch['input_ids'])
        inter_prompt = [
            self.enc_mlp(self.prompt[0]['prev_key'].to(self.device)),
            self.enc_mlp(self.prompt[0]['prev_value'].to(self.device)),
            self.dec_mlp(self.prompt[1]['prev_key'].to(self.device)),
            self.dec_mlp(self.prompt[1]['prev_value'].to(self.device)),
            self.cross_mlp(self.prompt[2]['prev_key'].to(self.device)),
            self.cross_mlp(self.prompt[2]['prev_value'].to(self.device))
        ]
        past_prompt = self.reformat(inter_prompt, bsz)

        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            decoder_input_ids=batch['decoder_input_ids'],
            labels=batch['labels'],
            past_prompt=past_prompt
        )
        valid_loss = outputs[0]
        self.log('valid_loss', valid_loss, on_step=True, on_epoch=True, logger=True)
        return {
            'valid_loss': valid_loss,
            'inter_prompt': inter_prompt
        }

    def validation_step_end(self, valid_output):
        if self.best_valid_loss > valid_output['valid_loss']:
            torch.save(valid_output['inter_prompt'], 'prefix_checkpoints/prefix_weights.pt')
            self.best_valid_loss = valid_output['valid_loss']

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.off_the_shelf_prefix is None:
            self.off_the_shelf_prefix = torch.load(self.off_the_shelf_prefix_path, map_location=self.device)
        bsz = len(batch['input_ids'])
        past_prompt = self.reformat(self.off_the_shelf_prefix, bsz, self.num_beams)

        # 这样不可以，最后一个batch会报错有零的会报错...还是得每个batch算一次prefix...
        '''if self.off_the_shelf_prefix is None:
            bsz = len(batch['input_ids'])
            num_beams = self.num_beams
            inter_prompt = torch.load(self.off_the_shelf_prefix_path, map_location=self.device)
            self.off_the_shelf_prefix = self.reformat(inter_prompt, bsz, num_beams)'''

        # 这里要注意一下，用了num_beams以后，bsz会变成bsz*num_beams，所以prefix的也要改...
        # 你问怎么改，我首先觉得可以在modeling_bart里面改，而且只用该decoder(decoder才用beam)
        # 上面一条其实也不好，更好的方法应该是改存法，存bsz为1的那种inter_prompt，然后调用reformat得到prefix
        # 这样不光存储的内存最优，也不用改模型里面的代码
        pred = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=100,
            num_beams=self.num_beams,
            length_penalty=1.0,
            past_prompt=past_prompt,
            # no_repeat_ngram_size=3
        )
        ref = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in pred]
        return ref

    def configure_optimizers(self):
        # to do: whether add a scheduler?
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    @staticmethod
    def reformat(prompt, bsz, num_beams=1):
        # 预测时因为beam search的关系，decoder的bsz会乘以num_beams，所以对应decoder的prefix也应该翻n倍bsz
        # 翻倍的情况出现在传入的num_beams覆盖掉1的时候
        # step 1: split hidden_dim(1024) into head_num(16)*head_dim(64)
        # [6][12, 1, 6, 1024] -> [6][12, 1, 6, 16, 64] -> [6][12, 1, 16, 6, 64]
        _, _, seq_len, _ = prompt[0].shape
        tmp_prompt_1 = [p.view(12, 1, seq_len, 16, 64).transpose(2, 3).contiguous()
                        for p in prompt]

        # step 2: expand the batch size from 1 to bsz
        # [6][12, 1, 16, 6, 64] -> [6][12, bsz, 16, 6, 64]
        tmp_prompt_2 = [p.repeat(1, bsz, 1, 1, 1) for p in tmp_prompt_1]

        # step 2.5: if num_beams != 1
        if num_beams > 1:
            for i in range(2, 6):
                tmp_prompt_2[i] = tmp_prompt_2[i].repeat(1, num_beams, 1, 1, 1)

        # step 3: reformat to the input of BART
        # [6][12, bsz, 16, 6, 64] -> [layer_num(12)][prompt_type][key/value][bsz, 16, 6, 64]
        past_prompt = []
        for i in range(12):
            past_prompt.append({
                'encoder_prompt': {
                    'prev_key': tmp_prompt_2[0][i],
                    'prev_value': tmp_prompt_2[1][i]
                },
                'decoder_prompt': {
                    'prev_key': tmp_prompt_2[2][i],
                    'prev_value': tmp_prompt_2[3][i]
                },
                'cross_attention_prompt': {
                    'prev_key': tmp_prompt_2[4][i],
                    'prev_value': tmp_prompt_2[5][i]
                }
            })
        return past_prompt
