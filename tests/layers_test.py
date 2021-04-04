import unittest

import numpy as np
from tplinker.layers import (ConcatHandshaking, DistanceEmbedding,
                             TaggingProjector)


class LayersTest(unittest.TestCase):

    def test_concat_handshaking(self):
        l = ConcatHandshaking(hidden_size=5)
        hidden = np.zeros([1, 3, 6], dtype=np.float)
        output = l(hidden)
        print(f"output shape: {output.shape}")

    def test_tagging_projector(self):
        l = TaggingProjector(24, 3)
        hidden = np.zeros([1, 6, 8], dtype=np.float)
        output = l(hidden)
        print(f"output shape: {output.shape}")

    def test_distance_embedding(self):
        l = DistanceEmbedding(100, 8)
        hidden = np.zeros([1, 6, 8], dtype=np.float)
        output = l(hidden)
        print(f"output shape: {output.shape}")
        print(output)


if __name__ == "__main__":
    unittest.main()
