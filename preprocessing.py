from vae2 import VAE
from database_management import Database
import ast
import numpy as np
from keras.utils import pad_sequences
from keras.models import Model
import tensorflow as tf
import typing


class Preprocessing(Database):
    def __init__(self,
                 max_input_length: int = 5,
                 max_input_text_length: int = 100,
                 max_output_length: int = 20,
                 database_path: str = "./data.db",
                 table: str = "patterns"
                 ) -> None:
        super().__init__(database_path, table)

        self.max_input_length = max_input_length
        self.max_input_text_length = max_input_text_length
        self.max_output_length = max_output_length

        self.dataset = self.get_table()

        self.outputs = [i[1] for i in self.dataset]
        self.matches = [list(ast.literal_eval(i[2])) for i in self.dataset]
        self.rejections = [list(ast.literal_eval(i[3])) for i in self.dataset]

        self.encoded_outputs = pad_sequences([self.encode_text(text) for text in self.outputs],
                                             maxlen=self.max_output_length,
                                             dtype="float",
                                             padding="post",
                                             truncating="post",
                                             value=0.)

        self.encoded_matches = pad_sequences([self.encode_texts(texts) for texts in self.matches],
                                             maxlen=self.max_input_length,
                                             dtype="float",
                                             padding="post",
                                             truncating="post",
                                             value=0.)

        self.encoded_rejections = pad_sequences([self.encode_texts(texts) for texts in self.rejections],
                                                maxlen=self.max_input_length,
                                                dtype="float",
                                                padding="post",
                                                truncating="post",
                                                value=0.)

        self.inputs = [self.encoded_matches, self.encoded_rejections]

        self.model = None

        print(self.encoded_outputs.shape, self.encoded_matches.shape,
              self.encoded_rejections.shape)

    def encode_text(self, text: str) -> list:
        return [ord(char) for char in text]

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return pad_sequences([self.encode_text(text) for text in texts],
                             maxlen=self.max_input_text_length,
                             dtype="float",
                             padding="post",
                             truncating="post",
                             value=0.)

    def compile(self, input_dim: int = 2, latent_dim: int = 5, optimizer: str = "adam"):
        self.model = VAE(input_dim, latent_dim)
        self.model.compile(optimizer=optimizer, loss=self.model.kl_reconstruction_loss)

    def train(self, epochs=10, batch_size=12):
        history = self.train_helper(
            self.inputs, self.encoded_outputs, epochs, batch_size)
        self.model.save_weights("model.h5")
        return history

    def train_helper(self, x, y, epochs, batch_size):
        history = self.model.fit(x, y, epochs=epochs, batch_size=batch_size)
        return history


p = Preprocessing()
p.compile()
p.train(batch_size=4)
