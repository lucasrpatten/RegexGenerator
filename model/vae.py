from keras.models import Model
from keras.layers import Input, Layer, Dense, Reshape, Bidirectional, LSTM, Masking, Flatten, Concatenate
from keras.losses import binary_crossentropy
import keras.backend as K
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

max_output_length = 20
max_input_length = 5
max_input_text_length = 100


class Encoder(Layer):
    def __init__(self, latent_dim, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.latent_dim = latent_dim
        self.hidden1 = LSTM(64, return_sequences=True,
                            go_backwards=True)
        self.hidden2 = LSTM(128, return_sequences=True,
                            go_backwards=True)
        self.hidden3 = LSTM(256, return_sequences=False,
                            go_backwards=True)
        self.flatten = Flatten()
        self.out = Dense(self.latent_dim)

    def compute_mask(self, inputs, mask=None):
        new_mask = K.not_equal(inputs, 0.0)
        if mask is not None:
            new_mask = mask * new_mask
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            'latent_dim': self.latent_dim
        })
        return config

    def call(self, inputs, mask=None):
        if mask is not None:
            inputs *= K.cast_to_floatx(mask)
        hidden = self.hidden1(inputs, mask=mask)
        hidden = self.hidden2(hidden, mask=mask)
        hidden = self.hidden3(hidden, mask=mask)
        hidden = self.flatten(hidden)
        z_mean = self.out(hidden)
        z_log_var = self.out(hidden)
        return z_mean, z_log_var


kl_loss = 0


class Sampling(Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


latent_dim = 20
match_encoder = Encoder(latent_dim=20)
rejection_encoder = Encoder(latent_dim=20)


encoder_matches_input = Input(
    shape=(max_input_length, max_input_text_length))
encoder_rejections_input = Input(
    shape=(max_input_length, max_input_text_length))
match_z_mean, match_z_log_var = match_encoder(encoder_matches_input)
rejection_z_mean, rejection_z_log_var = rejection_encoder(
    encoder_rejections_input)
z_mean = Concatenate(axis=1)([match_z_mean, rejection_z_mean])
z_log_var = Concatenate(axis=1)([match_z_log_var, rejection_z_log_var])

sampling = Sampling()([z_mean, z_log_var])
encoder_output = Concatenate(axis=1)([z_mean, z_log_var])
encoder = Model([encoder_matches_input, encoder_rejections_input],
                encoder_output, name="encoder")
encoder.summary()

decoder_input = Input(shape=(latent_dim*4), name="decoder_input")
hidden = Dense(512, activation="relu")(decoder_input)
hidden = Dense(256, activation="sigmoid")(hidden)
hidden = Dense(128, activation="relu")(hidden)
hidden = Dense(64, activation="relu")(hidden)
decoder_output = Dense(max_output_length, activation="linear")(hidden)
decoder = Model(decoder_input, decoder_output, name="decoder")
decoder.summary()

matches_input = Input(
    shape=(max_input_length, max_input_text_length), name="matches")
rejections_input = Input(
    shape=(max_input_length, max_input_text_length), name="rejections")

encoded_input = encoder([matches_input, rejections_input])
decoded_regex = decoder(encoded_input)
autoencoder = Model(
    [matches_input, rejections_input], decoded_regex, name="vae")
autoencoder.summary()
autoencoder.compile(optimizer="adam", loss=lambda true, pred: kl_reconstruction_loss(true, pred))

def kl_reconstruction_loss(truth, pred):
    reconstruction_loss = binary_crossentropy(truth, pred)
    return K.mean(reconstruction_loss + kl_loss)
