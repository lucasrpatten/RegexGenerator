from keras.models import Model
from keras.layers import Input, Dense, Reshape, Lambda, Layer, BatchNormalization, Bidirectional, LSTM, Masking, Flatten, Concatenate, Embedding
from keras.losses import binary_crossentropy
import keras.backend as K
import numpy as np
import tensorflow as tf


class Sampling(Model):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, trainable=True, name="sampling", dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

    def call(self, inputs):
        mu, sigma = inputs
        batch = K.shape(mu)[0]
        dim = K.shape(mu)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return mu + K.exp(sigma / 2) * epsilon


class Encoder(Model):
    def __init__(self, latent_dim=2, trainable=True, name="encoder", dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name="encoder", dtype=dtype, dynamic=dynamic, **kwargs)
        self.latent_dim = latent_dim
        self.masking = Masking(mask_value=0., name="masking")
        self.hidden1 = Bidirectional(LSTM(64, return_sequences=True, name="bidirectional_lstm_1"))
        self.hidden2 = Bidirectional(LSTM(128, return_sequences=True, name="bidirectional_lstm_2"))
        self.hidden3 = Bidirectional(LSTM(256, return_sequences=False, name="bidirectional_lstm_3"))
        self.normalization = BatchNormalization()
        self.flatten = Flatten()
        self.dense = Dense(512, activation="relu")
        self.mu = Dense(self.latent_dim, name="latent_mu")
        self.sigma = Dense(self.latent_dim, name="latent_sigma")

    def call(self, inputs):
        masked = self.masking(inputs)
        hidden = self.hidden1(masked)
        hidden = self.hidden2(hidden)
        hidden = self.hidden3(hidden)
        normalized = self.normalization(hidden)
        flattened = self.flatten(normalized)
        dense = self.dense(flattened)
        dense_normalized = self.normalization(dense)
        mu = self.mu(dense_normalized)
        sigma = self.sigma(dense_normalized)
            # rnn_shape = K.int_shape(normalized)
        return mu, sigma

class Decoder(Model):
    def __init__(self, latent_dim=2, max_output_len=20, trainable=True, name="decoder", dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        tf.config.run_functions_eagerly(True)
        self.latent_dim = latent_dim
        self.max_output_len = max_output_len
        self.hidden1 = LSTM(256, return_sequences=True)
        self.hidden2 = LSTM(128, return_sequences=True)
        self.hidden3 = LSTM(64, return_sequences=True)
        self.masking = Masking(mask_value=0.)
        self.out = Dense(self.max_output_len, activation='linear')
        self.embedding_layer = Embedding(input_dim=3, output_dim=2)

    def call(self, inputs):
        input1 = inputs[0].numpy()
        input2 = inputs[1].numpy()
        concatenated = np.concatenate([input1, input2], axis=-1)
        arr3d = np.expand_dims(concatenated, axis=1)
        arr_tensor = tf.constant(arr3d)
        hidden = self.hidden1(arr_tensor)
        hidden = self.hidden2(hidden)
        hidden = self.hidden3(hidden)
        masked = self.masking(hidden)
        output = self.out(masked)
        return output

class VAE(Model):
    def __init__(
        self,
        input_dim,
        latent_dim,
        max_output_len=20,
        beta=1.0,
        name="variational_autoencoder",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.max_output_len = max_output_len
        self.mu = None
        self.sigma = None
        self.z = None
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.latent_dim, self.max_output_len)
        self.sampling = Sampling()


    def kl_reconstruction_loss(self, truth, pred):
        reconstruction_loss = binary_crossentropy(
            K.flatten(truth), K.flatten(pred))

        kl_loss = 1 + self.sigma - K.square(self.mu) - K.exp(self.sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        return K.mean(reconstruction_loss + kl_loss)

    def call(self, inputs):
        concatenated = Concatenate(axis=1)(inputs)
        self.mu, self.sigma = self.encoder(concatenated)
        z = self.sampling([self.mu, self.sigma])
        predicted_output = self.decoder([z, self.mu, self.sigma])
        return predicted_output
