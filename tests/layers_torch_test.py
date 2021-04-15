import unittest

import numpy as np
import torch
from tplinker.layers_torch import (ConcatHandshaking, DistanceEmbedding,
                                   GloveEmbedding, TaggingProjector, TPLinker)


class LayersTest(unittest.TestCase):

    def test_glove_embedding(self):
        words = []
        with open('data/bert-base-cased/vocab.txt', mode='rt', encoding='utf-8') as fin:
            for line in fin:
                words.append(line.rstrip('\n'))
        vocab = {}
        for idx, token in enumerate(words):
            vocab[idx] = token
        ge = GloveEmbedding('data/glove_300_nyt.emb', vocab=vocab, embedding_size=300)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.int64)
        output = ge(input_ids)
        self.assertEqual([1, 8, 300], list(output.size()))

    def test_distance_embedding(self):
        de = DistanceEmbedding()
        hidden = np.zeros([1, 6, 8], dtype=np.float)
        output = de(torch.tensor(hidden))
        print(f"output shape: {output.size()}")
        self.assertEqual([1, 21, 768], list(output.size()))

    def test_concat_handshaking(self):
        cs = ConcatHandshaking(768)
        hidden = torch.zeros([1, 6, 768], dtype=torch.float32)
        output = cs(hidden)
        print(f"output shape: {output.size()}")
        self.assertEqual([1, 21, 768], list(output.size()))

    def test_tagging_projector(self):
        tp = TaggingProjector(768, 24)
        hidden = torch.zeros([1, 21, 768], dtype=torch.float32)
        output = tp(hidden)
        print(f"output shape: {output.size()}")
        self.assertEqual([1, 24, 21, 3], list(output.size()))

    def test_tplinker(self):
        tp = TPLinker(768, 24, add_dist_embedding=True)
        hidden = torch.zeros([1, 6, 768], dtype=torch.float32)
        h2t, h2h, t2t = tp(hidden)
        self.assertEqual([1, 21, 2], list(h2t.size()))
        self.assertEqual([1, 24, 21, 3], list(h2h.size()))
        self.assertEqual([1, 24, 21, 3], list(t2t.size()))


if __name__ == "__main__":
    unittest.main()
