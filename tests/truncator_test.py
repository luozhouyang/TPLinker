import json
import unittest

from tplinker.truncator import BertExampleTruncator
from transformers import BertTokenizerFast


class TruncatorTest(unittest.TestCase):

    def _read_examples(self, tokenizer, nums=1, min_sequence_length=100, **kwargs):
        examples = []
        with open('data/tplinker/bert/valid_data.jsonl', mode='rt', encoding='utf8') as fin:
            for line in fin:
                e = json.loads(line)
                tokens = tokenizer.tokenize(e['text'])
                if len(tokens) < min_sequence_length:
                    continue
                examples.append(e)
                # if len(examples) == 17:
                #     tokens = tokenizer.tokenize(e['text'])
                #     print(tokens)
                if len(examples) == nums:
                    break
        return examples

    def _create_truncator(self):
        tokenizer = BertTokenizerFast.from_pretrained(
            'data/bert-base-cased', add_special_tokens=False, do_lower_case=False)
        truncator = BertExampleTruncator(tokenizer, max_sequence_length=100)
        return truncator, tokenizer

    def test_bert_truncator(self):
        truncator, tokenizer = self._create_truncator()
        examples = self._read_examples(tokenizer, nums=17, min_sequence_length=100)
        truncated_examples = truncator.truncate(example=examples[-1])
        print('original example: ', examples[-1])
        for e in truncated_examples:
            print()
            print(e)

    def test_bert_truncator_example(self):
        example = {
            'text': 'Besides Mr. Stanley and Mr. Fugate , they include Richard Andrews , the former homeland security adviser to Gov. Arnold Schwarzenegger of California ; Ellen M. Gordon , former homeland security adviser in Iowa ; Dale W. Shipley of Ohio and Eric Tolbert of North Carolina , two former top FEMA officials who also served as the top emergency managers in their home states ; and Bruce P. Baughman , the president of the National Emergency Management Association , as well as the top disaster planning official in Alabama .', 'id': 'valid_1816',
            'relation_list': [{'subject': 'Arnold Schwarzenegger', 'object': 'California', 'subj_char_span': [113, 134], 'obj_char_span': [138, 148], 'predicate': '/people/person/place_lived', 'subj_tok_span': [24, 30], 'obj_tok_span': [31, 32]}, {'subject': 'Arnold Schwarzenegger', 'object': 'California', 'subj_char_span': [113, 134], 'obj_char_span': [138, 148], 'predicate': '/business/person/company', 'subj_tok_span': [24, 30], 'obj_tok_span': [31, 32]}],
            'entity_list': [{'text': 'Arnold Schwarzenegger', 'type': 'DEFAULT', 'char_span': [113, 134], 'tok_span': [24, 30]}, {'text': 'California', 'type': 'DEFAULT', 'char_span': [138, 148], 'tok_span': [31, 32]}, {'text': 'Arnold Schwarzenegger', 'type': 'DEFAULT', 'char_span': [113, 134], 'tok_span': [24, 30]}, {'text': 'California', 'type': 'DEFAULT', 'char_span': [138, 148], 'tok_span': [31, 32]}]
        }
        truncator, _ = self._create_truncator()
        outputs = truncator.truncate(example)
        for o in outputs:
            print()
            print(o)


if __name__ == "__main__":
    unittest.main()
