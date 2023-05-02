from database_management import Database
import ast
import numpy as np
from model.generator import RegexGenerator
import json
import os


class DBLoader(Database):
    def __init__(self, max_output_length=20, database_path="./data.db", table="patterns"):
        super().__init__(database_path=database_path, table=table)
        self.max_output_length = max_output_length
        self.dataset, self.ouputs, self.matches, self.rejections = [[None]] * 4

    def load_data(self):
        self.dataset = self.get_table()
        self.outputs = [i[1] for i in self.dataset]
        self.matches = [list(ast.literal_eval(i[2])) for i in self.dataset]
        self.rejections = [list(ast.literal_eval(i[3])) for i in self.dataset]


class Preprocessing(DBLoader):
    def __init__(self,
                 max_input_length: int = 5,
                 max_input_text_length: int = 100,
                 max_output_length: int = 20,
                 database_path: str = "./data.db",
                 table: str = "patterns"
                 ):
        super().__init__(max_output_length=max_output_length,
                         database_path=database_path, table=table)
        self.max_input_length = max_input_length
        self.max_input_text_length = max_input_text_length
        self.max_output_length = max_output_length

    def encode_text(self, text: str, maxlen: int) -> np.ndarray:
        padded = np.zeros(maxlen, dtype=np.float32)
        for i, char in enumerate(text[:maxlen]):
            padded[i] = ord(char)/128
        return padded

    def encode_texts(self, texts: list[str], maxtextlen: int, maxarrlen: int) -> np.ndarray:
        padded = np.pad(texts, (0, maxarrlen - len(texts)),
                        'constant', constant_values="")
        encoded_texts = np.array([self.encode_text(
            text, maxlen=maxtextlen) for text in padded])
        return encoded_texts

    def preprocess_database(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.load_data()
        encoded_matches = np.array([self.encode_texts(
            i, self.max_input_text_length, self.max_input_length) for i in self.matches])
        encoded_rejections = np.array([self.encode_texts(
            i, self.max_input_text_length, self.max_input_length) for i in self.rejections])
        encoded_outputs = np.array(
            [self.encode_text(i, self.max_output_length) for i in self.outputs])
        return encoded_matches, encoded_rejections, encoded_outputs


def train():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    p = Preprocessing()
    encoded_matches, encoded_rejections, encoded_outputs = p.preprocess_database()
    # print(encoded_matches.shape)
    # input_dim: int = 2
    latent_dim: int = 5
    epochs, batch_size = 40, 3
    model = RegexGenerator()
    model.compile(optimizer="adam", loss="mse")
    history = model.fit([encoded_matches, encoded_rejections],
                        encoded_outputs, batch_size, epochs)
    model.save_weights("model.h5")
    with open("build_config.json", "w") as f:
        json.dump(model.get_config(), f)

    r = r"[a-zA-Z]+\d{2}"
    matches = list(('abAB12', 'cdCD34', 'efEF56', 'ghGH78'))
    rejections = list(('abc', 'def', 'ghi', 'jkl'))
    p = Preprocessing()
    matches = p.encode_texts(matches, 100, 5)
    rejections = p.encode_texts(rejections, 100, 5)

    matches = matches.reshape((1, 5, 100))
    rejections = rejections.reshape((1, 5, 100))

    response = model([matches, rejections])
    print(response)
    print([chr(int(abs(i)*128)) for i in response[0]])


train()
