import json
import os

import torch
from preprocessor.truncator import BertExampleTruncator
from tplinker.scheme import HandshakingTaggingEncoder, TagMapping
from transformers import BertTokenizerFast


class TPLinkerBertDataset(torch.utils.data.Dataset):

    def __init__(self,
                 input_files,
                 pretrained_bert_path,
                 rel2id_path,
                 max_sequence_length=100,
                 batch_size=64,
                 **kwargs):
        self.bert_truncator = BertExampleTruncator(pretrained_bert_path, max_sequence_length=max_sequence_length)
        self.tokenizer = self.bert_truncator.tokenizer
        self.examples = self._read_input_files(input_files)

        with open(rel2id_path, mode='rt', encoding='utf-8') as fin:
            rel2id = json.load(fin)
        self.tag_mapping = TagMapping(rel2id)
        self.encoder = HandshakingTaggingEncoder(self.tag_mapping)
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.index = 0

    def _read_input_files(self, input_files):
        if isinstance(input_files, str):
            input_files = [input_files]
        all_examples = []
        for f in input_files:
            with open(f, mode='rt', encoding='utf-8') as fin:
                for line in fin:
                    example = json.loads(line)
                    examples = self.bert_truncator.truncate(example)
                    if examples:
                        all_examples.extend(examples)
        return all_examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        codes = self.tokenizer.encode_plus(
            example['text'],
            return_offsets_mapping=True,
            add_special_tokens=False,
            max_length=self.max_sequence_length,
            padding='max_length')

        h2t, h2h, t2t = self.encoder.encode([example], max_sequence_length=self.max_sequence_length)

        result = {
            'input_ids': torch.tensor(codes['input_ids']),
            'attention_mask': torch.tensor(codes['attention_mask']),
            'token_type_ids': torch.tensor(codes['token_type_ids']),
            'h2t': torch.tensor(h2t[0]),
            'h2h': torch.tensor(h2h[0]),
            't2t': torch.tensor(t2t[0])
        }
        return result
