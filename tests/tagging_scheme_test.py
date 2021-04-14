import json
import unittest

import numpy as np
from tplinker.tagging_scheme import HandshakingTaggingEncoder, TagMapping


class SchemeTest(unittest.TestCase):

    def _build_encoder(self):
        with open('data/tplinker/bert/rel2id.json', mode='rt', encoding='utf-8') as fin:
            rel2id = json.load(fin)
        tm = TagMapping(relation2id=rel2id)
        encoder = HandshakingTaggingEncoder(tag_mapping=tm)
        return encoder

    def test_handshaking_tagging_encoder(self):
        encoder = self._build_encoder()

        count = 0
        with open('data/tplinker/bert/train_data.jsonl', mode='rt', encoding='utf-8') as fin:
            for line in fin:
                example = json.loads(line)
                print()
                print('example: {}'.format(example))
                h2t, h2h, t2t = encoder.encode(example, max_sequence_length=100)
                print(f'h2t shape: {h2t.shape}, h2h shape: {h2h.shape}, t2t shape: {t2t.shape}')
                # print(h2t)
                # print(np.sum(h2t))
                # print(h2h)
                # print(t2t)

                count += 1
                if count == 100:
                    break

    def test_handshaking_tagging_ecoder_example(self):
        example = {
            'text': "-LRB- Dunning -RRB- NEXT WAVE FESTIVAL : NATIONAL BALLET OF CHINA -LRB- Tuesday through Thursday -RRB- The company will perform '' Raise the Red Lantern , '' which tells the story of a young concubine in 1920 's China with a fusion of ballet , modern dance and traditional Chinese dance , set to music performed on Western and Chinese instruments and directed by Zhang Yimou",
            'entity_list': [{'text': 'Zhang Yimou', 'type': 'DEFAULT', 'tok_span': [96, 100], 'char_span': [363, 374]}, {'text': 'China', 'type': 'DEFAULT', 'tok_span': [70, 71], 'char_span': [212, 217]}],
            'relation_list': [{'subj_tok_span': [96, 100], 'obj_tok_span': [70, 71], 'subj_char_span': [363, 374], 'obj_char_span': [212, 217], 'subject': 'Zhang Yimou', 'object': 'China', 'predicate': '/people/person/nationality'}], 'token_offset': 0, 'char_offset': 0,
            'offset_mapping': [[0, 1], [1, 2], [2, 4], [4, 5], [6, 10], [10, 13], [14, 15], [15, 16], [16, 18], [18, 19], [20, 22], [22, 24], [25, 27], [27, 29], [30, 31], [31, 33], [33, 35], [35, 37], [37, 38], [39, 40], [41, 42], [42, 44], [44, 47], [47, 49], [50, 52], [52, 54], [54, 56], [57, 59], [60, 62], [62, 64], [64, 65], [66, 67], [67, 68], [68, 70], [70, 71], [72, 79], [80, 87], [88, 96], [97, 98], [98, 99], [99, 101], [101, 102], [103, 106], [107, 114], [115, 119], [120, 127], [128, 129], [129, 130], [131, 134], [134, 136], [137, 140], [141, 144], [145, 152], [153, 154], [155, 156], [156, 157], [158, 163], [164, 169], [170, 173], [174, 179], [180, 182], [183, 184], [185, 190], [191, 194], [194, 196], [196, 200], [201, 203], [204, 208], [209, 210], [210, 211], [212, 217], [218, 222], [223, 224], [225, 231], [232, 234], [235, 241], [242, 243], [244, 250], [251, 256], [257, 260], [261, 272], [273, 280], [281, 286], [287, 288], [289, 292], [293, 295], [296, 301], [302, 311], [312, 314], [315, 322], [323, 326], [327, 334], [335, 346], [347, 350], [351, 359], [360, 362], [363, 368], [369, 371], [371, 373], [373, 374]]
        }

        encoder = self._build_encoder()

        outputs = encoder.encode(example)
        print(outputs)


if __name__ == "__main__":
    unittest.main()
