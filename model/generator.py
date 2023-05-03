"""
Contains the generator model
"""

import typing
from keras.layers import LSTM, Concatenate, Dense, Layer
from keras.models import Model


class Hidden(Layer):
    """Hidden layer containing an RNN made of LSTM layers

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
        self.hidden1 = LSTM(512, return_sequences=True, go_backwards=True)
        self.hidden2 = LSTM(256, return_sequences=True, go_backwards=True)
        self.hidden3 = LSTM(128, return_sequences=True, go_backwards=True)
        self.hidden4 = LSTM(64, return_sequences=True, go_backwards=True)
        self.hidden5 = LSTM(32)

    def call(self, inputs):
        # x = tf.transpose(inputs, perm=[0, 2, 1])
        hidden = self.hidden1(inputs)
        hidden = self.hidden2(hidden)
        hidden = self.hidden3(hidden)
        hidden = self.hidden4(hidden)
        hidden = self.hidden5(hidden)
        return hidden


class RegexGenerator(Model):
    """Model that generates regular expressions

    Args:
        max_output_len (int, optional): Maximum output pattern length. Defaults to 20.

        name (str, optional): Name of the model. Defaults to "regex_generator".
    """

    def __init__(self, max_output_len=20, name="regex_generator", *args, **kwargs):
        super(RegexGenerator, self).__init__(name=name, *args, **kwargs)
        self.max_output_len = max_output_len
        self.matches = Hidden(name="matches")
        self.rejections = Hidden(name="rejections")
        self.concatenate = Concatenate()
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
        matches = self.matches(matches)
        rejections = self.rejections(rejections)
        combined = self.concatenate([matches, rejections])
        hidden = self.dense(combined)
        hidden = self.dense1(hidden)
        out = self.output_layer(hidden)
        return out
