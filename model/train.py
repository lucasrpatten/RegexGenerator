"""
    Contains function to train the model
"""

import json
from generator import RegexGenerator
from preprocessing import Preprocessing
import tensorflow as tf

def custom_loss(y_true, y_pred):
    int_true = tf.multiply(128.0, y_true)
    int_pred = tf.multiply(128.0, y_pred)
    squared_difference = tf.square(int_true - int_pred)
    return tf.reduce_mean(squared_difference, axis=-1)

def train():
    """ Function to train the model from the dataset
    """
    process = Preprocessing(database_path="./model/data.db")
    encoded_matches, encoded_rejections, encoded_outputs = process.preprocess_database()
    epochs, batch_size = 32, 128
    model = RegexGenerator()

    model.compile(optimizer="adam", loss=custom_loss)
    model.fit([encoded_matches, encoded_rejections],
              encoded_outputs, batch_size, epochs)
    model.save_weights("model.h5")
    with open("build_config.json", "w", encoding="utf8") as file:
        json.dump(model.get_config(), file)

    _ = r"[a-zA-Z]+\d{2}"
    matches = list(('abAB12', 'cdCD34', 'efEF56', 'ghGH78'))
    rejections = list(('abc', 'def', 'ghi', 'jkl'))
    matches = process.encode_texts(matches, 100, 10)
    rejections = process.encode_texts(rejections, 100, 10)

    matches = matches.reshape((1, 10, 100))
    rejections = rejections.reshape((1, 10, 100))

    response = model([matches, rejections])
    print(response)
    print([chr(int(abs(i)*128)) for i in response[0]])

train()
