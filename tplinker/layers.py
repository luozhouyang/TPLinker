import math

import numpy as np
import tensorflow as tf


class DistanceEmbedding(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

    def build(self, input_shape):

        def _initializer(shape, dtype=None):
            vocab_size, embedding_size = shape[0], shape[1]
            matrix = np.zeros([vocab_size, embedding_size])
            for d in range(self.vocab_size):
                for i in range(self.embedding_size):
                    if i % 2 == 0:
                        matrix[d][i] = math.sin(d / 10000**(i / self.embedding_size))
                    else:
                        matrix[d][i] = math.cos(d / 10000**((i - 1) / self.embedding_size))
            return matrix

        self.dist_embedding = self.add_weight(
            'distance_embedding_table',
            shape=(self.vocab_size, self.embedding_size),
            dtype=self.dtype,
            initializer=_initializer,
            trainable=False)

        super().build(input_shape)

    def call(self, hidden_state, **kwargs):
        batch_size, seq_len = tf.shape(hidden_state)[0], tf.shape(hidden_state)[1]
        segs = []
        for index in range(seq_len, 0, -1):
            segs.append(self.dist_embedding[:index, :])
        segs = tf.concat(segs, axis=0)
        return tf.tile(segs[None, :, :], [batch_size, 1, 1])


class TaggingProjector(tf.keras.layers.Layer):

    def __init__(self, num_relations, num_tags=3, **kwargs):
        super().__init__(**kwargs)
        self.fc_layers = [tf.keras.layers.Dense(num_tags) for _ in range(num_relations)]

    def call(self, hidden_states, **kwargs):
        """The projection for relation hidden states.

        Args:
            hidden_state: Tensor, shape is (batch_size, seq_len, depth)

        Returns:
            outputs: Tensor, shape is (batch_size, num_relations, seq_len, num_tags)
        """
        outputs = []
        for i, fc in enumerate(self.fc_layers):
            output = fc(hidden_states)
            outputs.append(output)
        return tf.stack(outputs, axis=1)


class ConcatHandshaking(tf.keras.layers.Layer):

    def __init__(self, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.fc = tf.keras.layers.Dense(hidden_size)

    def call(self, hidden, **kwargs):
        """
        Args:
            hidden: Tensor, shape (batch_size, seq_len, depth)

        Returns:
            handshaking_hiddens: Tensor, shape (batch_size, 1+2+...+seq_len, hidden_size)
        """
        seq_len = tf.shape(hidden)[1]
        handshaking_hiddens = []
        for index in range(seq_len):
            _hidden = hidden[:, index, :]
            _hidden = _hidden[:, None, :]
            repeat_hidden = tf.tile(_hidden, [1, seq_len - index, 1])
            handshaking_hidden = tf.concat([repeat_hidden, hidden[:, index:, :]], axis=-1)
            handshaking_hidden = tf.tanh(self.fc(handshaking_hidden))
            handshaking_hiddens.append(handshaking_hidden)
        handshaking_hiddens = tf.concat(handshaking_hiddens, axis=1)
        return handshaking_hiddens

# TODO: implement conditional layer norm and layer norm handshaking


class ConditionalLayerNorm(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        pass


class LayerNormHandshaking(tf.keras.layers.Layer):

    def __init__(self, hidden_size, **kwargs):
        pass

    def call(self, inputs, **kwargs):
        pass
