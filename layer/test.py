from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np

from layer import attention
from layer import encodec


def test_mask():
    x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    print(attention.create_padding_mask(x))


def test_scaled_dot_product_attention():
    def print_out(q, k, v):
        temp_out, temp_attn = attention.scaled_dot_product_attention(
            q, k, v, None)
        print('Attention weights are:')
        print(temp_attn)
        print('Output is:')
        print(temp_out)

    np.set_printoptions(suppress=True)
    temp_k = tf.constant([[10, 0, 0],
                          [0, 10, 0],
                          [0, 0, 10],
                          [0, 0, 10]], dtype=tf.float32)  # (4, 3)

    temp_v = tf.constant([[1, 0],
                          [10, 0],
                          [100, 5],
                          [1000, 6]], dtype=tf.float32)  # (4, 2)

    temp_q = tf.constant([[0, 0, 10], [0, 10, 0], [10, 10, 0]], dtype=tf.float32)  # (3, 3)
    print_out(temp_q, temp_k, temp_v)


def test_multi_head_attention():
    temp_mha = attention.MultiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(y, k=y, q=y, mask=None)
    print(out.shape, attn.shape)


def test_encoder_layer():
    sample_encoder_layer = encodec.EncoderLayer(512, 8, 2048)

    sample_encoder_layer_output = sample_encoder_layer(
        tf.random.uniform((64, 43, 512)), False, None)

    print(sample_encoder_layer_output.shape)  # [N L D]


def test_encoder():
    sample_encoder = encodec.Encoder(num_layers=2, d_model=512, num_heads=8,
                                     dff=2048, input_vocab_size=8500,
                                     maximum_position_encoding=10000)

    sample_encoder_output = sample_encoder(tf.random.uniform((64, 62)),
                                           training=False, mask=None)

    print(sample_encoder_output.shape)  # [N L D]


if __name__ == '__main__':
    # tf.compat.v1.enable_eager_execution()
    test_encoder()
