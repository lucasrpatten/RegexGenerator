from keras.models import Model
from keras.layers import Input, Layer, Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Reshape, Bidirectional, LSTM, Masking, Flatten, Concatenate
from keras.losses import binary_crossentropy
from keras.metrics import Mean
import keras
import keras.backend as K
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

max_output_length = 20
max_input_length = 5
max_input_text_length = 100


class Sampling(Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


latent_dim = 8
encoder_inputs = Input(shape=(max_input_text_length))
x = LSTM(64, return_sequences=True, go_backwards=True,
         pad_zeroes=True)(encoder_inputs)
x = LSTM(128, return_sequences=True, go_backwards=True, pad_zeroes=True)(x)
x = LSTM(256, return_sequences=True, go_backwards=True, pad_zeroes=True)(x)
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
z_mean = Dense(latent_dim, name="z_mean")(x)
z_log_var = Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = Input(shape=(latent_dim,))
x = Dense(1280, activation="relu")(latent_inputs)
x = Dense(640, activation="relu")(x)
x = LSTM(256, return_sequences=True)(x)
x = LSTM(128, return_sequences=True)(x)
decoder_outputs = Dense(max_input_text_length, activation="sigmoid")(x)
decoder = Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()


class VAE(Model):
    def __init__(self, encoder, decoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var -
                              tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# class Encoder(Layer):
#     def __init__(self, latent_dim, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
#         super().__init__(trainable, name, dtype, dynamic, **kwargs)
#         self.latent_dim = latent_dim
#         self.flatten = Flatten()
#         self.out = Dense(self.latent_dim)

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             'latent_dim': self.latent_dim
#         })
#         return config

#     def call(self, inputs):
#         hidden = self.flatten(inputs)
#         z_mean = self.out(hidden)
#         z_log_var = self.out(hidden)
#         return z_mean, z_log_var


# class Sampling(Layer):
#     def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
#         super().__init__(trainable, name, dtype, dynamic, **kwargs)

#     def call(self, inputs):
#         z_mean, z_log_var = inputs
#         batch = K.shape(z_mean)[0]
#         dim = K.int_shape(z_mean)[1]
#         epsilon = K.random_normal(shape=(batch, dim))
#         return z_mean + K.exp(0.5 * z_log_var) * epsilon


# latent_dim = 80

# encoder_matches_input = Input(
#     shape=(max_input_length, max_input_text_length))
# encoder_rejections_input = Input(
#     shape=(max_input_length, max_input_text_length))
# encoder_input = Concatenate(
#     axis=-1)([encoder_matches_input, encoder_rejections_input])
# z_mean, z_log_var = Encoder(latent_dim=latent_dim,
#                             name="encoder_layer")(encoder_input)
# sampling = Sampling()([z_mean, z_log_var])
# encoder = Model([encoder_matches_input, encoder_rejections_input],
#                 sampling, name="encoder")
# encoder.summary()

# decoder_input = Input(shape=(latent_dim), name="decoder_input")
# hidden = Dense(512, activation='relu')(decoder_input)
# hidden = Dense(256, activation="relu")(hidden)
# hidden = Dense(128, activation="relu")(hidden)
# hidden = Dense(64, activation="relu")(hidden)
# decoder_output = Dense(max_output_length, activation="linear")(hidden)
# decoder = Model(decoder_input, decoder_output, name="decoder")
# decoder.summary()

# matches_input = Input(
#     shape=(max_input_length, max_input_text_length), name="matches")
# rejections_input = Input(
#     shape=(max_input_length, max_input_text_length), name="rejections")

# encoded_input = encoder([matches_input, rejections_input])
# decoded_regex = decoder(encoded_input)
# autoencoder = Model(
#     [matches_input, rejections_input], decoded_regex, name="vae")
# autoencoder.summary()
# autoencoder.compile(optimizer="adam", loss=lambda true,
#                     pred: kl_reconstruction_loss(true, pred))


# def kl_reconstruction_loss(truth, pred):
#     truth = 128 * truth
#     pred = 128 * pred
#     reconstruction_loss = mean_absolute_error(truth, pred)
#     return reconstruction_loss
