from keras.models import Model
from keras.layers import Dense, LSTM, Layer, Concatenate


class Hidden(Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.hidden1 = LSTM(512, return_sequences=True, go_backwards=True)
        self.hidden2 = LSTM(256, return_sequences=True, go_backwards=True)
        self.hidden3 = LSTM(128, return_sequences=True, go_backwards=True)
        self.hidden4 = LSTM(64, return_sequences=True, go_backwards=True)
        self.hidden5 = LSTM(32)

    def call(self, inputs):
        # x = tf.transpose(inputs, perm=[0, 2, 1])
        x = self.hidden1(inputs)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        return x


class RegexGenerator(Model):
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
        x = self.dense(combined)
        x = self.dense1(x)
        x = self.output_layer(x)
        return x
