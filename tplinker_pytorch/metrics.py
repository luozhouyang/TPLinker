import json

import pytorch_lightning as pl
import torch
from torchmetrics import Metric

from tplinker_pytorch.tagging_scheme import (HandshakingTaggingDecoder,
                                             TagMapping)


class SampleAccuracy(Metric):

    def __init__(self):
        super().__init__()
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # shape: (batch_size, num_relations, 1+2+...+seq_len)
        preds_id = torch.argmax(preds, dim=-1)
        # shape: (batch_size, num_relations * (1+2+...+seq_len))
        preds_id = preds_id.view(preds_id.size()[0], -1)
        # shape: (batch_size, num_relations * (1+2+...+seq_len))
        target = target.view(target.size()[0], -1)
        # num of correct tags
        correct_tags = torch.sum(torch.eq(target, preds_id), dim=1)
        # num of correct samples
        correct_samples = torch.sum(
            torch.eq(correct_tags, torch.ones_like(correct_tags) * target.size()[-1]))

        self.correct += correct_samples
        self.total += target.size()[0]

    def compute(self):
        return self.correct / self.total


class _PRF(Metric):
    """Precision, Recall and F1 metric"""

    def __init__(self, rel2id_path, max_sequence_length=100, pattern='only_head_text', epsilon=1e-12):
        super().__init__()
        self.pattern = pattern
        self.max_sequence_length = max_sequence_length
        self.epsilon = epsilon

        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('goldnum', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('prednum', default=torch.tensor(0), dist_reduce_fx='sum')

        with open(rel2id_path, mode='rt', encoding='utf-8') as fin:
            rel2id = json.load(fin)
        tag_mapping = TagMapping(rel2id)
        self.decoder = HandshakingTaggingDecoder(tag_mapping)

    def update(self, preds, target):
        examples = [json.loads(e) for e in target['example']]
        pred_h2t, pred_h2h, pred_t2t = preds['h2t'], preds['h2h'], preds['t2t']

        for index in range(len(examples)):
            # print('process No.{} example'.format(index))
            self._count(examples[index], pred_h2t[index], pred_h2h[index], pred_t2t[index])

    def _count(self, example, h2t_pred, h2h_pred, t2t_pred):
        pred_relations = self.decoder.decode(
            example, h2t_pred, h2h_pred, t2t_pred,
            max_sequence_length=self.max_sequence_length)
        gold_relations = example['relation_list']

        pred_relations_set, gold_relations_set = self._parse_relations_set(pred_relations, gold_relations)
        for rel in pred_relations_set:
            if rel in gold_relations_set:
                self.correct += 1
        self.goldnum += len(gold_relations_set)
        self.prednum += len(pred_relations_set)

    def _parse_relations_set(self, pred_relations, gold_relations):
        if self.pattern == 'whole_span':
            gold_set = set(['{}-{}-{}-{}-{}'.format(
                rel['subj_tok_span'][0], rel['subj_tok_span'][1], rel['predicate'], rel['obj_tok_span'][0], rel['obj_tok_span'][1]
            ) for rel in gold_relations])
            pred_set = set(['{}-{}-{}-{}-{}'.format(
                rel['subj_tok_span'][0], rel['subj_tok_span'][1], rel['predicate'], rel['obj_tok_span'][0], rel['obj_tok_span'][1]
            ) for rel in pred_relations])
            return pred_set, gold_set
        if self.pattern == 'whole_text':
            gold_set = set([
                '{}-{}-{}'.format(rel['subject'], rel['predicate'], rel['object']) for rel in gold_relations
            ])
            pred_set = set([
                '{}-{}-{}'.format(rel['subject'], rel['predicate'], rel['object']) for rel in pred_relations
            ])
            return pred_set, gold_set
        if self.pattern == 'only_head_index':
            gold_set = set([
                '{}-{}-{}'.format(rel['subj_tok_span'][0], rel['predicate'], rel['obj_tok_span'][0]) for rel in gold_relations
            ])
            pred_set = set([
                '{}-{}-{}'.format(rel['subj_tok_span'][0], rel['predicate'], rel['obj_tok_span'][0]) for rel in pred_relations
            ])
            return pred_set, gold_set
        gold_set = set([
            '{}-{}-{}'.format(rel['subject'].split(' ')[0], rel['predicate'], rel['object'].split(' ')[0]) for rel in gold_relations
        ])
        pred_set = set([
            '{}-{}-{}'.format(rel['subject'].split(' ')[0], rel['predicate'], rel['object'].split(' ')[0]) for rel in pred_relations
        ])
        return pred_set, gold_set


class Precision(_PRF):

    def compute(self):
        return self.correct / (self.prednum + self.epsilon)


class Recall(_PRF):

    def compute(self):
        return self.correct / (self.goldnum + self.epsilon)


class F1(_PRF):

    def compute(self):
        precision = self.correct / (self.prednum + self.epsilon)
        recall = self.correct / (self.goldnum + self.epsilon)
        f1 = 2.0 * precision * recall / (precision + recall + self.epsilon)
        return f1
