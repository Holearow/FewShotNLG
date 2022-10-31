import torch
from transformers import (
    BartTokenizer,
    BartModel,
    BartForConditionalGeneration
)
from pytorch_lightning import LightningModule


class NaiveBartModel(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model_path = kwargs['model_path']
        self.tokenizer_path = kwargs['tokenizer_path']
        self.lr = kwargs['lr']

        self.model = BartForConditionalGeneration.from_pretrained(self.model_path)
        self.tokenizer = BartTokenizer.from_pretrained(self.tokenizer_path)

    def training_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            decoder_input_ids=batch['decoder_input_ids'],
            labels=batch['labels']
        )
        train_loss = outputs[0]
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            decoder_input_ids=batch['decoder_input_ids'],
            labels=batch['labels']
        )
        valid_loss = outputs[0]
        self.log('valid_loss', valid_loss, on_step=True, on_epoch=True, logger=True)
        return valid_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # 这样预测是有问题的，这样deocder的输入是shift_token_right的encoder input...
        # outputs = self.model(
        #     input_ids=batch['input_ids'],
        #     attention_mask=batch['attention_mask']
        # )
        # logits = outputs[0]
        # pred_ids = logits.argmax(dim=2)
        # pred_ids = pred_ids[:, :64].contiguous()
        # sentences = self.tokenizer.batch_decode(pred_ids)
        pred = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=100,
            num_beams=4,
            length_penalty=2.0,
            # no_repeat_ngram_size=3
        )
        ref = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in pred]
        return ref

    def configure_optimizers(self):
        # to do: whether add a scheduler?
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
