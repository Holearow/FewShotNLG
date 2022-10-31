import os
import torch
import pytorch_lightning as pl

from torch import nn
from torch.utils.data import DataLoader
from dataloader import WikiDataset
from transformers import (
    BartTokenizer,
    BartModel,
    BartForConditionalGeneration
)
from define_model import NaiveBartModel
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


tokenizer = BartTokenizer.from_pretrained('bart-large/')


def convert_to_features(batch):
    # get batch and tokenize
    table_batch = [s[0] for s in batch]
    ref_batch = [s[1] for s in batch]

    table_encodings = tokenizer.batch_encode_plus(
        table_batch,
        max_length=300,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    ref_encodings = tokenizer.batch_encode_plus(
        ref_batch,
        max_length=64,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    # deal with sequence format & <pad> token in decoder step
    pad_token_id = tokenizer.pad_token_id

    decoder_tmp = ref_encodings['input_ids']
    decoder_input_ids = decoder_tmp[:, :-1].contiguous()

    labels = decoder_tmp[:, 1:].clone()
    labels[decoder_tmp[:, 1:] == pad_token_id] = -100

    # return batch for training
    encodings = {
        'input_ids': table_encodings['input_ids'],
        'attention_mask': table_encodings['attention_mask'],
        'decoder_input_ids': decoder_input_ids,
        'labels': labels
    }
    return encodings


def main(dataset_args, trainer_args, model_args):
    # ensure reproducibility
    seed_everything(611, workers=True)

    # get arguments
    batch_size = dataset_args['bsz']
    domain = dataset_args['domain']
    data_availability = dataset_args['data_availability']

    # access to data
    train_set = WikiDataset(domain=domain, mode='train', data_availability=data_availability)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=convert_to_features)

    valid_set = WikiDataset(domain=domain, mode='valid', data_availability=data_availability)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=convert_to_features)

    test_set = WikiDataset(domain=domain, mode='test', data_availability=data_availability)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=convert_to_features)

    # initialize the model
    model = NaiveBartModel(**model_args)

    # initialize the trainer
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="valid_loss",
        mode="min",
        dirpath="./checkpoints/",
        filename='10-31-{epoch}-{step}-{val_loss:.2f}'
    )
    logger = TensorBoardLogger(save_dir='./logs/', version=1, name="lightning_logs")

    trainer = Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        # limit_predict_batches=2,
        **trainer_args
    )

    # training & predicting
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    predictions = trainer.predict(model, dataloaders=test_loader, return_predictions=True)
    with open('predictions/books_50_predictions.txt', 'w', encoding='UTF-8') as f:
        for prediction_batch in predictions:
            for pred in prediction_batch:
                f.write(pred + '\n')


if __name__ == '__main__':
    # define arguments
    dataset_args = {
        'root_path': './data_release/',
        'domain': 'books',
        'data_availability': False,
        'bsz': 10
    }

    trainer_args = {
        'accelerator': 'gpu',
        'devices': '1',
        'deterministic': True,
        'max_steps': 600
    }

    model_args = {
        'lr': 5e-5,
        'model_path': 'bart-large/',
        'tokenizer_path': 'bart-large/'
    }

    main(dataset_args, trainer_args, model_args)
