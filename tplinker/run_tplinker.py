import argparse
import json
import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import (Callback, EarlyStopping,
                                         ModelCheckpoint)

from .dataset import TPLinkerBertDataset
from .metrics import F1, Precision, Recall, SampleAccuracy
from .models_torch import TPLinkerBert
from .tagging_scheme import HandshakingTaggingDecoder, TagMapping


def _compute_loss(y_pred, y_true):
    y_pred = y_pred.view(-1, y_pred.size()[-1])
    y_true = y_true.view(-1)
    return F.cross_entropy(y_pred, y_true)


class ONNXModelExport(Callback):

    def __init__(self, export_dir, model_name='model'):
        super().__init__()
        self.export_dir = export_dir
        if not os.path.exists(self.export_dir) or not os.path.isdir(self.export_dir):
            os.makedirs(self.export_dir)
        self.model_name = model_name

    def on_train_epoch_end(self, trainer, pl_module: pl.LightningModule, outputs):
        filename = os.path.join(
            self.export_dir,
            '{}-epoch-{}.onnx'.format(self.model_name, trainer.current_epoch))

        device = pl_module.device
        input_sample = (
            torch.ones((1, 100)).long().to(device),
            torch.ones((1, 100)).long().to(device),
            torch.zeros((1, 100)).long().to(device))
        # pl_module.to_onnx(filename, input_sample, export_params=True)
        torch.onnx.export(
            pl_module, input_sample, filename,
            opset_version=10,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)


class TPLinkerLightning(pl.LightningModule):

    def __init__(self, model, rel2id_path, max_sequence_length=100, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.h2t_acc = SampleAccuracy()
        self.h2h_acc = SampleAccuracy()
        self.t2t_acc = SampleAccuracy()
        self.train_precision = Precision()
        self.train_recll = Recall()
        self.train_f1 = F1()

        with open(rel2id_path, mode='rt', encoding='utf-8') as fin:
            rel2id = json.load(fin)
        tag_mapping = TagMapping(rel2id)
        self.decoder = HandshakingTaggingDecoder(tag_mapping)
        self.max_sequence_length = max_sequence_length

    def forward(self, input_ids, attention_mask, token_type_ids, **kwargs):
        h2t, h2h, t2t = self.model(input_ids, attention_mask, token_type_ids)
        return h2t, h2h, t2t

    def train_dataloader(self):
        train_dataset = TPLinkerBertDataset(
            input_files=['data/tplinker/bert/train_data.jsonl'],
            pretrained_bert_path='data/bert-base-cased',
            rel2id_path='data/tplinker/bert/rel2id.json')
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        return train_dataloader

    def val_dataloader(self):
        valid_dataset = TPLinkerBertDataset(
            input_files=['data/tplinker/bert/valid_data.jsonl'],
            pretrained_bert_path='data/bert-base-cased',
            rel2id_path='data/tplinker/bert/rel2id.json')
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False)
        return valid_dataloader

    def training_step(self, batch, index, **kwargs):
        input_ids, attn_mask, type_ids = batch['input_ids'], batch['attention_mask'], batch['token_type_ids']
        h2t_pred, h2h_pred, t2t_pred = self.model(input_ids, attn_mask, type_ids)
        h2t_loss = _compute_loss(h2t_pred, batch['h2t'])
        h2h_loss = _compute_loss(h2h_pred, batch['h2h'])
        t2t_loss = _compute_loss(t2t_pred, batch['t2t'])
        total_loss = 0.333 * h2t_loss + 0.333 * h2h_loss + 0.333 * t2t_loss
        logs = {'h2t_loss': h2t_loss, 'h2h_loss': h2h_loss, 't2t_loss': t2t_loss}
        logs.update({
            'h2t_acc': self.h2t_acc(h2t_pred, batch['h2t']),
            'h2h_acc': self.h2h_acc(h2h_pred, batch['h2h']),
            't2t_acc': self.t2t_acc(t2t_pred, batch['t2t']),
        })

        # decoding predictions takes long time in early epochs, skip to decrease training time
        if self.trainer.current_epoch > 0:
            examples = [json.loads(e) for e in batch['example']]
            pred_relations = self.decoder.batch_decode(
                examples,
                h2t_pred, h2h_pred, t2t_pred,
                max_sequence_length=self.max_sequence_length)
            # print('num of pred relations: {}'.format([len(x) for x in pred_relations]))
            gold_relations = [e['relation_list'] for e in examples]
            # TODO: Fixed metrics
            logs.update({
                'precision': self.train_precision(pred_relations, gold_relations),
                'recall': self.train_recll(pred_relations, gold_relations),
                'f1': self.train_f1(pred_relations, gold_relations),
            })
        self.log_dict(logs, prog_bar=True, on_step=True, on_epoch=False)
        return total_loss

    def training_epoch_end(self, outputs):
        print('correct: {}, prednum: {}, goldnum: {}'.format(
            self.train_precision.correct,
            self.train_precision.prednum,
            self.train_precision.goldnum))

    def validation_step(self, batch, index, **kwargs):
        input_ids, attn_mask, type_ids = batch['input_ids'], batch['attention_mask'], batch['token_type_ids']
        h2t_pred, h2h_pred, t2t_pred = self.model(input_ids, attn_mask, type_ids)
        h2t_loss = _compute_loss(h2t_pred, batch['h2t'])
        h2h_loss = _compute_loss(h2h_pred, batch['h2h'])
        t2t_loss = _compute_loss(t2t_pred, batch['t2t'])
        total_loss = 0.333 * h2t_loss + 0.333 * h2h_loss + 0.333 * t2t_loss
        logs = {'val_h2t_loss': h2t_loss, 'val_h2h_loss': h2h_loss, 'val_t2t_loss': t2t_loss, 'val_loss': total_loss}
        self.log_dict(logs, prog_bar=False, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=3e-5)
        # schedule = torch.optim.lr_scheduler.CosineAnnealingRestarts(opt,)
        return opt


def create_trainer(model_path='model/', gpus=0, **kwargs):
    trainer = pl.Trainer(
        gpus=gpus,
        default_root_dir=model_path,
        callbacks=[
            ModelCheckpoint(
                dirpath=os.path.join(model_path, 'ckpt'),
                filename='tplinker-bert-{epoch}-{step}-{val_loss:.2f}',
                monitor='val_loss',
                save_top_k=kwargs.get('save_top_k', 5),
                mode='min'
            ),
            EarlyStopping(monitor='val_loss'),
            ONNXModelExport(export_dir=os.path.join(model_path, 'onnx'), model_name='tplinker-bert')
        ],
        max_epochs=kwargs.get('max_epochs', 10),
    )
    return trainer


def create_bert_model(pretrained_bert_path, num_relations, add_distance_embedding=False):
    model = TPLinkerBert(
        bert_model_path=pretrained_bert_path,
        num_relations=num_relations,
        add_distance_embedding=add_distance_embedding)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--pretrained_bert_path', default='data/bert-base-cased')
    parser.add_argument('--num_relations', default=24)
    parser.add_argument('--add_distance_embedding', default=False)
    parser.add_argument('--model_path', default='data/model/tplinker-bert/v0')
    parser.add_argument('--save_top_k', default=5)
    parser.add_argument('--max_epochs', default=10)
    parser.add_argument('--max_sequence_length', default=100)

    args, _ = parser.parse_known_args()

    module = TPLinkerLightning(
        model=create_bert_model(
            pretrained_bert_path=args.pretrained_bert_path,
            num_relations=args.num_relations,
            add_distance_embedding=args.add_distance_embedding),
        rel2id_path='data/tplinker/bert/rel2id.json',
        max_sequence_length=args.max_sequence_length)
    trainer = create_trainer(
        model_path=args.model_path,
        gpus=args.gpus,
        save_top_k=args.save_top_k,
        max_epochs=args.max_epochs,
    )
    trainer.fit(module)
