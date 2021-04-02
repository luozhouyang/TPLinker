import logging
from collections import namedtuple

import numpy as np


class TagMapping:

    def __init__(self, relation2id, **kwargs):
        super().__init__()
        # relation mapping
        self.relation2id = relation2id
        self.id2relation = {v: k for k, v in self.relation2id.items()}

        # tag2id mapping: entity head to entity tail
        self.head2tail_tag2id = {
            "O": 0,
            "ENT-H2T": 1,  # entity head to entity tail
        }
        self.head2tail_id2tag = {v: k for k, v in self.head2tail_tag2id.items()}

        # tag2id mapping: entity1 head to entity2 head
        self.head2head_tag2id = {
            "O": 0,
            "REL-SH2OH": 1,  # subject head to object head
            "REL-OH2SH": 2,  # object head to subject head
        }
        self.head2head_id2tag = {v: k for k, v in self.head2head_tag2id.items()}

        # tag2id mapping: entity1 tail to entity2 tail
        self.tail2tail_tag2id = {
            "O": 0,
            "REL-ST2OT": 1,  # subject tail to object tail
            "REL-OT2ST": 2,  # object tail to subject tail
        }
        self.tail2tail_id2tag = {v: k for k, v in self.tail2tail_tag2id.items()}

    def relation_id(self, key):
        return self.relation2id.get(key, -1)

    def h2t_id(self, key):
        return self.head2tail_tag2id.get(key, 0)

    def h2t_tag(self, _id):
        return self.head2tail_id2tag.get(_id, 'O')

    def h2h_id(self, key):
        return self.head2head_tag2id.get(key, 0)

    def h2h_tag(self, _id):
        return self.head2head_id2tag.get(_id, 'O')

    def t2t_id(self, key):
        return self.tail2tail_tag2id.get(key, 0)

    def t2t_tag(self, _id):
        return self.tail2tail_id2tag.get(_id, 'O')


# p -> point to head of entity, q -> point to tail of entity, tagid -> id of h2t tag
Head2TailItem = namedtuple('Head2TailItem', ['p', 'q', 'tagid'])
# p -> point to head of entity1, q -> point to head of entity2
# relid -> relation_id between entity1 and entity2, tagid -> id of h2h tag
Head2HeadItem = namedtuple('Head2HeadItem', ['relid', 'p', 'q', 'tagid'])
# p -> point to tail of entity1, q -> point to tail of entity2
# relid -> relation_id between entity1 and entity2, tagid -> id of h2h tag
Tail2TailItem = namedtuple('Tail2TailItem', ['relid', 'p', 'q', 'tagid'])


class HandshakingTaggingEncoder:

    def __init__(self, tag_mapping: TagMapping, **kwargs):
        super().__init__()
        self.tag_mapping = tag_mapping

    def encode(self, instances, max_sequence_length=100, **kwargs):
        index_matrix = self._build_index_matrix(max_sequence_length=max_sequence_length, **kwargs)
        flatten_length = max_sequence_length * (max_sequence_length + 1) // 2
        batch_h2t_spots, batch_h2h_spots, batch_t2t_spots = [], [], []
        for instance in instances:
            h2t_spots, h2h_spots, t2t_spots = self._collect_spots(instance, **kwargs)
            batch_h2t_spots.append(h2t_spots)
            batch_h2h_spots.append(h2h_spots)
            batch_t2t_spots.append(t2t_spots)
        h2t_tagging = self._encode_head2tail(batch_h2t_spots, index_matrix=index_matrix, sequence_length=flatten_length)
        h2h_tagging = self._encode_head2head(batch_h2h_spots, index_matrix=index_matrix, sequence_length=flatten_length)
        t2t_tagging = self._encode_tail2tail(batch_t2t_spots, index_matrix=index_matrix, sequence_length=flatten_length)
        return h2t_tagging, h2h_tagging, t2t_tagging

    def _collect_spots(self, instance, **kwargs):
        h2t_spots, h2h_spots, t2t_spots = [], [], []
        # TODO: 考虑不在relation_list中的entity
        for relation in instance['relation_list']:
            subject_span = relation['subj_tok_span']
            object_span = relation['obj_tok_span']

            # add head-to-tail spot
            h2t_i0 = Head2TailItem(p=subject_span[0], q=subject_span[1]-1, tagid=self.tag_mapping.h2t_id('ENT-H2T'))
            h2t_spots.append(h2t_i0)
            h2t_i1 = Head2TailItem(p=object_span[0], q=object_span[1]-1, tagid=self.tag_mapping.h2t_id('ENT-H2T'))
            h2t_spots.append(h2t_i1)
            # convert relation to id
            relid = self.tag_mapping.relation_id(relation['predicate'])
            # add head-to-head spot
            p = subject_span[0] if subject_span[0] <= object_span[0] else object_span[0]
            q = object_span[0] if subject_span[0] <= object_span[0] else subject_span[0]
            k = 'REL-SH2OH' if subject_span[0] <= object_span[0] else 'REL-OH2SH'
            h2h_item = Head2HeadItem(relid=relid, p=p, q=q, tagid=self.tag_mapping.h2h_id(k))
            h2h_spots.append(h2h_item)

            # add tail-to-tail spot
            p = subject_span[1] if subject_span[1] <= object_span[1] else object_span[1]
            q = object_span[1] if subject_span[1] <= object_span[1] else subject_span[1]
            k = 'REL-ST2OT' if subject_span[1] <= object_span[1] else 'REL-OT2ST'
            t2t_item = Tail2TailItem(relid=relid, p=p, q=q, tagid=self.tag_mapping.t2t_id(k))
            t2t_spots.append(t2t_item)

        return h2t_spots, h2h_spots, t2t_spots

    def _encode_head2tail(self, batch_h2t_spots, index_matrix, sequence_length, **kwargs):
        batch_tagging_sequence = np.zeros([len(batch_h2t_spots), sequence_length], dtype=np.int)
        for batch_id, h2t_spots in enumerate(batch_h2t_spots):
            for item in h2t_spots:
                index = index_matrix[item.p][item.q]
                batch_tagging_sequence[batch_id][index] = item.tagid
        return batch_tagging_sequence

    def _encode_head2head(self, batch_h2h_spots, index_matrix, sequence_length, **kwargs):
        num_relations = len(self.tag_mapping.relation2id)
        # shape (num_relations, sequence_length)
        batch_tagging_sequence = np.zeros([len(batch_h2h_spots), num_relations, sequence_length], dtype=np.int)
        for batch_id, h2h_spots in enumerate(batch_h2h_spots):
            for item in h2h_spots:
                index = index_matrix[item.p][item.q]
                batch_tagging_sequence[batch_id][item.relid][index] = item.tagid
        return batch_tagging_sequence

    def _encode_tail2tail(self, batch_t2t_spots, index_matrix, sequence_length, **kwargs):
        num_relations = len(self.tag_mapping.relation2id)
        # shape (num_relations, sequence_length)
        batch_tagging_sequence = np.zeros([len(batch_t2t_spots), num_relations, sequence_length], dtype=np.int)
        for batch_id, t2t_spots in enumerate(batch_t2t_spots):
            for item in t2t_spots:
                index = index_matrix[item.p][item.q]
                batch_tagging_sequence[batch_id][item.relid][index] = item.tagid
        return batch_tagging_sequence

    def _build_index_matrix(self, max_sequence_length=100, **kwargs):
        # e.g [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        pairs = [(i, j) for i in range(max_sequence_length) for j in list(range(max_sequence_length))[i:]]
        # shape: (max_sequence_length, max_sequence_length)
        matrix = [[0 for i in range(max_sequence_length)] for j in range(max_sequence_length)]
        for index, values in enumerate(pairs):
            matrix[values[0]][values[1]] = index
        return matrix


class HandshakingTaggingDecoder:

    def __init__(self, tag_mapping: TagMapping, max_sequence_length: int, **kwargs):
        super().__init__()
        self.tag_mapping = tag_mapping
        self.max_sequence_length = max_sequence_length

    def decode(self, inputs, **kwargs):
        pass
