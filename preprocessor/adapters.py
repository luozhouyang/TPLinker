import abc
import json
import logging
import os
import re
from collections import namedtuple

from transformers import BertTokenizerFast

Entity = namedtuple('Entity', ['text', 'span', 'type'])
Relation = namedtuple('Relation', ['subject', 'object', 'subject_span', 'object_span', 'predict'])
Example = namedtuple('Example', ['id', 'text', 'relations', 'entities'])


class AbstractDatasetAdapter(abc.ABC):

    @abc.abstractmethod
    def adapte(self, input_file, output_file, **kwargs):
        raise NotImplementedError()


class NYTBertAdapter(AbstractDatasetAdapter):

    def __init__(self, pretrained_bert_path, add_special_tokens=False, do_lower_case=False, **kwargs):
        super().__init__()
        self.do_lower_case = do_lower_case
        self.tokenizer = BertTokenizerFast.from_pretrained(
            pretrained_bert_path, add_special_tokens=add_special_tokens, do_lower_case=do_lower_case)

    def adapte(self, input_file, output_file, **kwargs):
        with open(output_file, mode='wt', encoding='utf-8') as fout, \
                open(input_file, mode='rt', encoding='utf-8') as fin:
            count = 0
            for line in fin:
                data = json.loads(line)
                example = self._adapte_example(data)
                # example.pop('offset', None)
                # json.dump(example, fout, ensure_ascii=False)
                # fout.write('\n')
                # print(example)
                self._validate_example(example)
                count += 1
                if count == kwargs.get('limit', -1):
                    break

    def _adapte_example(self, data):
        text = data['sentText']
        codes = self.tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)
        example = {
            'text': text,
            'tokens': self.tokenizer.convert_ids_to_tokens(codes['input_ids']),
            'ids': codes['input_ids'],
            'offset': codes['offset_mapping']
        }
        self._adapte_entities(data, example)
        # TODO: finishe relations adaption
        # self._adapte_relations(data, example)
        return example

    def _adapte_entities(self, data, example):
        text = data['sentText']
        entity_list = []
        for e in data['entityMentions']:
            for m in re.finditer(re.escape(e['text']), text):
                char_span_start, char_span_end = m.span()[0], m.span()[1]
                # prev character is number
                if char_span_start > 0 and re.match('\d', text[char_span_start - 1]):
                    continue
                # next character is number
                if char_span_end < len(text) and re.match('\d', text[char_span_end]):
                    continue
                # get token span by char span
                token_span_start, token_span_end = self._parse_token_span(example, char_span_start, char_span_end)
                if not token_span_start or not token_span_end:
                    print('invalid token span for entity: {}, regex match span: {}'.format(e, m.span()))
                    continue
                entity_list.append({
                    'text': e['text'],
                    'type': e['label'],
                    'token_span': [token_span_start, token_span_end],
                    'char_span': [char_span_start, char_span_end]
                })
        example.update({
            'entity_list': entity_list
        })

    def _adapte_relations(self, data, example):
        entities = {e['text']: e for e in example['entity_list']}
        relations_list = []
        for relation in data['relationMentions']:
            relations_list.append({
                'subject': None,
                'object': None,
                'predict': None
            })

    def _parse_token_span(self, example, start, end):
        token_start, token_end = None, None
        for idx, (token, offset) in enumerate(zip(example['tokens'], example['offset'])):
            if offset[0] == start and end == offset[1]:
                return idx, idx + 1
            if offset[0] == start:
                token_start = idx
            if end == offset[1]:
                token_end = idx
            if token_start is not None and token_end is not None:
                return token_start, token_end + 1
        return token_start, token_end

    def _validate_example(self, example):
        tokens = example['tokens']
        text = example['text']
        for entity in example['entity_list']:
            start, end = entity['token_span']
            print()
            print('tokens subsequence: {}'.format(tokens[start:end]))
            print('entity text: {}'.format(entity['text']))
            char_start, char_end = entity['char_span']
            print('origin text: {}'.format(text[char_start:char_end]))
