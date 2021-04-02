import json
import unittest

from tplinker.scheme import HandshakingTaggingEncoder, TagMapping


class SchemeTest(unittest.TestCase):

    def test_handshaking_tagging_encoder(self):
        with open('data/tplinker/bert/rel2id.json', mode='rt', encoding='utf-8') as fin:
            rel2id = json.load(fin)
        tm = TagMapping(relation2id=rel2id)
        encoder = HandshakingTaggingEncoder(tag_mapping=tm, max_sequence_length=100)

        count = 0
        with open('data/tplinker/bert/train_data.jsonl', mode='rt', encoding='utf-8') as fin:
            for line in fin:
                instance = json.loads(line)
                h2t, h2h, t2t = encoder.encode([instance])
                print()
                print(f'h2t shape: {h2t.shape}, h2h shape: {h2h.shape}, t2t shape: {t2t.shape}')
                print(h2t)
                print(h2h)
                print(t2t)

                count += 1
                if count == 5:
                    break


if __name__ == "__main__":
    unittest.main()
