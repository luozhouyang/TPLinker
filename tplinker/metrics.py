import json

import pytorch_lightning as pl
import torch
from torchmetrics import Metric

from tplinker.tagging_scheme import HandshakingTaggingDecoder, TagMapping


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

    def __init__(self, pattern='only_head_text', epsilon=1e-12):
        super().__init__()
        self.pattern = pattern
        self.epsilon = epsilon

        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('goldnum', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('prednum', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, pred_relations, gold_relations):
        for pred, gold in zip(pred_relations, gold_relations):
            pred_set, gold_set = self._parse_relations_set(pred, gold)
            for rel in pred_set:
                if rel in gold_set:
                    self.correct += 1
            self.prednum += len(pred_set)
            self.goldnum += len(gold_set)
        # print('metric states: correct={}, prednum={}, goldnum={}'.format(self.correct, self.prednum, self.goldnum))

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
