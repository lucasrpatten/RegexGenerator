"""
    Contains function to train the model
"""

import json
from generator import RegexGenerator
from preprocessing import Preprocessing


def train():
    """ Function to train the model from the dataset
    """
    process = Preprocessing(database_path="./model/data.db")
    encoded_matches, encoded_rejections, encoded_outputs = process.preprocess_database()
    epochs, batch_size = 20, 8
    model = RegexGenerator()
    model.compile(optimizer="adam", loss="mse")
    model.fit([encoded_matches, encoded_rejections],
              encoded_outputs, batch_size, epochs)
    model.save_weights("model.h5")
    with open("build_config.json", "w", encoding="utf8") as file:
        json.dump(model.get_config(), file)

    _ = r"[a-zA-Z]+\d{2}"
    matches = list(('abAB12', 'cdCD34', 'efEF56', 'ghGH78'))
    rejections = list(('abc', 'def', 'ghi', 'jkl'))
    matches = process.encode_texts(matches, 100, 5)
    rejections = process.encode_texts(rejections, 100, 5)

    matches = matches.reshape((1, 5, 100))
    rejections = rejections.reshape((1, 5, 100))

    response = model([matches, rejections])
    print(response)
    print([chr(int(abs(i)*128)) for i in response[0]])

train()
