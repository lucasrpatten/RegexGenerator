from keras.layers import Dense, LSTM, Flatten, Bidirectional, TimeDistributed, Reshape, Concatenate, Masking, RepeatVector
from keras.models import Model
from keras.layers import Layer
import keras.backend as K
import tensorflow as tf


class Sampling(Layer):
    def __init__(self, trainable=True, name="sampling", dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        z = z_mean + K.exp(0.5 * z_log_var) * epsilon
        return z


class Encoder(Layer):
    def __init__(self, input_dim, latent_dim, trainable=True, name="encoder", dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.bidirectional_lstm_1 = Bidirectional(
            LSTM(256, return_sequences=True))
        self.bidirectional_lstm_2 = Bidirectional(
            LSTM(128, return_sequences=True))
        self.bidirectional_lstm_3 = Bidirectional(
            LSTM(64, return_sequences=False))
        self.mean = Dense(self.latent_dim, name="mean", activation='linear')
        self.log_var = Dense(latent_dim, name="log_var", activation='linear')
        self.sampling = Sampling()

    def call(self, inputs):
        concatenated = Concatenate(axis=1)([inputs[0], inputs[1]])
        hidden1 = self.bidirectional_lstm_1(concatenated)
        hidden2 = self.bidirectional_lstm_2(hidden1)
        hidden3 = self.bidirectional_lstm_3(hidden2)
        mean, log_var = self.mean(hidden3), self.log_var(hidden3)
        z = self.sampling([mean, log_var])
        return z, mean, log_var


class Decoder(Layer):
    def __init__(self, input_dim, latent_dim, trainable=True, name="decoder", dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dense_1 = Dense(64, name="dense_1", activation='relu')
        self.repeat_vector = RepeatVector(input_dim)
        self.lstm_1 = LSTM(128, name="lstm_1", return_sequences=True)
        self.lstm_2 = LSTM(256, name="lstm_2", return_sequences=True)
        self.dense_2 = TimeDistributed(
            Dense(input_dim, name="dense_2", activation="sigmoid"), name="time_distributed")

    def call(self, inputs):
        hidden_1 = self.dense_1(inputs)
        repeat = self.repeat_vector(hidden_1)
        hidden_2 = self.lstm_1(repeat)
        hidden_3 = self.lstm_2(hidden_2)
        output = self.dense_2(hidden_3)
        return output


class VAE(Model):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def encode(self, inputs):
        z, mean, log_var = self.encoder(inputs)
        return z, mean, log_var

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mean, log_var):
        # To enable backpropagation
        eps = K.random_normal(shape=mean.shape)
        return mean + K.exp(0.5 * log_var) * eps

    def call(self, inputs):
        _, mean, log_var = self.encode(inputs)
        z = self.reparameterize(mean, log_var)
        reconstructed = self.decode(z)
        self.add_loss(self.calc_loss(inputs, reconstructed, mean, log_var))
        return reconstructed, mean, log_var

    def calc_loss(self, inputs, reconstructed, mean, log_var):
        reconstruction_loss = tf.reduce_mean(K.square(inputs - reconstructed))
        kl_divergence_loss = -0.5 * tf.reduce_mean(1 + log_var - K.square(mean) - K.exp(log_var))
        total_loss = reconstruction_loss + kl_divergence_loss
        return total_loss