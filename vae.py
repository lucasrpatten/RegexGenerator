from keras.layers import Dense, LSTM, Flatten, Bidirectional, Reshape, Concatenate, Masking, RepeatVector
from keras.models import Model
from keras.layers import Layer
import keras.backend as K


class Encoder(Layer):
    def __init__(self, input_dim, latent_dim, name="encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.bidirectional_lstm_1 = Bidirectional(
            LSTM(256, return_sequences=True))
        self.bidirectional_lstm_2 = Bidirectional(
            LSTM(128, return_sequences=True))
        self.bidirectional_lstm_3 = Bidirectional(
            LSTM(64, return_sequences=False))
        self.z_mean = Dense(self.latent_dim, activation='linear')
        self.z_log_var = Dense(self.latent_dim, activation='linear')

    def process(self, inputs):
        masked_input = Masking(mask_value=0.)(inputs)
        hidden1 = self.bidirectional_lstm_1(masked_input)
        hidden2 = self.bidirectional_lstm_2(hidden1)
        hidden3 = self.bidirectional_lstm_3(hidden2)
        return hidden3

    def call(self, match, reject):
        matches = self.process(match)
        rejections = self.process(reject)
        concatenated = Concatenate(axis=1)([matches, rejections])
        flattened = Flatten()(concatenated)
        z_mean = self.z_mean(flattened)
        z_log_var = self.z_log_var(flattened)
        return z_mean, z_log_var


class Decoder(Layer):
    def __init__(self, input_dim, latent_dim, max_output_len, name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.max_output_len = max_output_len
        self.bidirectional_lstm_1 = Bidirectional(
            LSTM(64, return_sequences=True))
        self.bidirectional_lstm_2 = Bidirectional(
            LSTM(128, return_sequences=True))
        self.bidirectional_lstm_3 = Bidirectional(
            LSTM(256, return_sequences=True))
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs):
        repeated_z = RepeatVector(self.max_output_len)(inputs[0])
        hidden1 = self.bidirectional_lstm_1(repeated_z)
        hidden2 = self.bidirectional_lstm_2(hidden1)
        hidden3 = self.bidirectional_lstm_3(hidden2)
        output = self.output_layer(hidden3)
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
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.max_output_len = max_output_len
        self.beta = beta
        self.encoder = Encoder(self.input_dim, self.latent_dim)
        self.decoder = Decoder(
            self.input_dim, self.latent_dim, self.max_output_len)

    def sample_z(self, z_mean, z_log_var):
        epsilon = K.random_normal(
            shape=(K.shape(z_mean)[0], self.latent_dim), mean=0., stddev=1.)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs[0], inputs[1])
        z = self.sample_z(z_mean, z_log_var)
        reconstruction = self.decoder([z, inputs[2]])
        return reconstruction

    def vae_loss(self, y_true, y_pred):
        reconstruction_loss = K.binary_crossentropy(y_true, y_pred)
        reconstruction_loss = K.sum(reconstruction_loss, axis=-1)
        kl_loss = -0.5 * K.sum(1 + self.encoder.z_log_var - K.square(
            self.encoder.z_mean) - K.exp(self.encoder.z_log_var), axis=-1)

        vae_loss = K.mean(reconstruction_loss + self.beta * kl_loss)
        return vae_loss
