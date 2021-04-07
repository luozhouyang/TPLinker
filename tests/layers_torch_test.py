import unittest

import numpy as np
import torch
from tplinker_pytorch.layers_torch import (ConcatHandshaking,
                                           DistanceEmbedding, TaggingProjector,
                                           TPLinker)


class LayersTest(unittest.TestCase):

    def test_distance_embedding(self):
        de = DistanceEmbedding()
        hidden = np.zeros([1, 6, 8], dtype=np.float)
        output = de(torch.tensor(hidden))
        print(f"output shape: {output.size()}")
        self.assertEquals([1, 21, 768], list(output.size()))

    def test_concat_handshaking(self):
        cs = ConcatHandshaking(768)
        hidden = torch.zeros([1, 6, 768], dtype=torch.float32)
        output = cs(hidden)
        print(f"output shape: {output.size()}")
        self.assertEquals([1, 21, 768], list(output.size()))

    def test_tagging_projector(self):
        tp = TaggingProjector(768, 24)
        hidden = torch.zeros([1, 21, 768], dtype=torch.float32)
        output = tp(hidden)
        print(f"output shape: {output.size()}")
        self.assertEquals([1, 24, 21, 3], list(output.size()))

    def test_tplinker(self):
        tp = TPLinker(768, 24, add_dist_embedding=True)
        hidden = torch.zeros([1, 6, 768], dtype=torch.float32)
        h2t, h2h, t2t = tp(hidden)
        self.assertEqual([1, 21, 2], list(h2t.size()))
        self.assertEqual([1, 24, 21, 3], list(h2h.size()))
        self.assertEqual([1, 24, 21, 3], list(t2t.size()))


if __name__ == "__main__":
    unittest.main()
