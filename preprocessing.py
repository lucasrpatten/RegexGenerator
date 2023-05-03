"""
File containing preprocessing and DBLoader classes
"""
import ast
import numpy as np

from database_management import Database


class DBLoader(Database):
    """Loads the database

    Args:
        max_output_length (int, optional): Maximum regex output length. Defaults to 20.

        database_path (str, optional): Where the database is saved. Defaults to "./data.db".

        table (str, optional): Name of the database table. Defaults to "patterns".
    """

    def __init__(
            self,
            max_output_length: int = 20,
            database_path: str = "./data.db",
            table: str = "patterns"
    ):
        super().__init__(database_path=database_path, table=table)
        self.max_output_length = max_output_length
        self.dataset, self.outputs, self.matches, self.rejections = [[""]] * 4

    def load_data(self):
        """Loads the data from the database
        """
        self.dataset = self.get_table()
        self.outputs = [i[1] for i in self.dataset]
        self.matches = [list(ast.literal_eval(i[2])) for i in self.dataset]
        self.rejections = [list(ast.literal_eval(i[3])) for i in self.dataset]


class Preprocessing(DBLoader):
    """Contains preprocessing methods and functions

    Args:
        max_input_length (int, optional):Maximum number of match/rejection inputs (each).
        Defaults to 5.

        max_input_text_length (int, optional): Maximum length of each input. Defaults to 100.

        max_output_length (int, optional): Maximum length of the output regex. Defaults to 20.

        database_path (str, optional): Where the database is stored. Defaults to "./data.db".

        table (str, optional): What table in the database to pull data from. Defaults to "patterns".
    """

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
        """Encodes an individual text

        Args:
            text (str): Text to encode
            maxlen (int): Max allowed length of text

        Returns:
            np.ndarray: padded and encoded text
        """
        padded = np.zeros(maxlen, dtype=np.float32)
        for i, char in enumerate(text[:maxlen]):
            padded[i] = ord(char)/128
        return padded

    def encode_texts(self, texts: list[str], maxtextlen: int, maxarrlen: int) -> np.ndarray:
        """Encodes multiple texts

        Args:
            texts (list[str]): List of texts to encode
            maxtextlen (int): maximum allowed length of each text
            maxarrlen (int): maximum length of input list

        Returns:
            np.ndarray: padded and encoded texts list
        """
        padded = np.pad(texts, (0, maxarrlen - len(texts)),
                        'constant', constant_values="")
        encoded_texts = np.array([self.encode_text(
            text, maxlen=maxtextlen) for text in padded])
        return encoded_texts

    def preprocess_database(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocesses the database

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Encoded Match Texts,
            Encoded Rejection Texts, Encoded Output Patterns
        """
        self.load_data()
        encoded_matches = np.array([self.encode_texts(
            i, self.max_input_text_length, self.max_input_length) for i in self.matches])
        encoded_rejections = np.array([self.encode_texts(
            i, self.max_input_text_length, self.max_input_length) for i in self.rejections])
        encoded_outputs = np.array(
            [self.encode_text(i, self.max_output_length) for i in self.outputs])
        return encoded_matches, encoded_rejections, encoded_outputs
