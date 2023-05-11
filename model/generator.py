"""
Contains the generator model
"""

import re
import typing
from keras.layers import LSTM, Dense, Layer, MultiHeadAttention, Masking, Bidirectional, Embedding
from keras.models import Model
import tensorflow as tf
import keras.backend as K


class FeatureExtraction(Layer):
    """Encoding layer containing an RNN made of LSTM layers. Performs feauture extraction by encoding the input into a latent space

    Args:
        trainable (bool, optional): Are this layers weights trainable. Defaults to True.

        name (str | None, optional): Layer name. Defaults to None.

        dtype (typing.Any, optional): dtype of layers computation and weights. Defaults to None.

        dynamic (bool, optional): True if only run eagerly. Defaults to False.
    """

    def __init__(
            self,
            trainable: bool = True,
            name: str | None = None,
            dtype: typing.Any = None,
            dynamic: bool = False,
            **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.masking_layer = Masking(mask_value=0.)
        self.hidden1 = Bidirectional(LSTM(512, return_sequences=True))
        self.hidden2 = Bidirectional(LSTM(256, return_sequences=True))
        self.hidden3 = Bidirectional(LSTM(128))

    def call(self, inputs):
        # x = tf.transpose(inputs, perm=[0, 2, 1])
        masked_inputs = self.masking_layer(inputs)
        hidden = self.hidden1(masked_inputs)
        hidden = self.hidden2(hidden)
        hidden = self.hidden3(hidden)
        return hidden


class RegexGenerator(Model):
    """Model that generates regular expressions

    Args:
        max_output_len (int, optional): Maximum output pattern length. Defaults to 50.

        name (str, optional): Name of the model. Defaults to "regex_generator".
    """

    def __init__(self, max_output_len=50, name="regex_generator", *args, **kwargs):
        super(RegexGenerator, self).__init__(name=name, *args, **kwargs)
        self.max_output_len = max_output_len\
        # TODO: Test if using seperate encoders or one encoder is more effective
        self.features = FeatureExtraction(name="feature_extraction_encoder")
        # I hypothesise that it is more effective to use two seperate encoders, but much more
        # Practical and computationally friendly to use one encoder
        self.attention = MultiHeadAttention(
            num_heads=2, key_dim=12, attention_axes=1)
        self.dense = Dense(512, activation="relu")
        self.dense1 = Dense(256, activation="relu")
        self.output_layer = Dense(max_output_len, activation="tanh")

    def get_config(self):
        return {
            "max_output_len": self.max_output_len
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs, training=None, mask=None):
        matches, rejections = inputs
        matches = self.features(matches)
        rejections = self.features(rejections)
        attention = self.attention(matches, rejections)
        hidden = self.dense(attention)
        hidden = self.dense1(hidden)
        out = self.output_layer(hidden)
        return out
