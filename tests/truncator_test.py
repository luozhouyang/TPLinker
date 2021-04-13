import json
import unittest

from preprocessor.truncator import BertExampleTruncator
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

    def test_bert_truncator(self):
        tokenizer = BertTokenizerFast.from_pretrained(
            'data/bert-base-cased', add_special_tokens=False, do_lower_case=False)
        truncator = BertExampleTruncator(tokenizer, max_sequence_length=100)
        examples = self._read_examples(tokenizer, nums=17, min_sequence_length=100)
        truncated_examples = truncator.truncate(example=examples[-1])
        print('original example: ', examples[-1])
        for e in truncated_examples:
            print()
            print(e)


if __name__ == "__main__":
    unittest.main()
