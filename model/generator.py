import time
import keras
from keras.models import Model
from keras.layers import Input, Dense, LSTM
import matplotlib.pyplot as plt

class RegexGenerator(Model):
    def __init__(self, name="regex_generator", *args, **kwargs):
        super(RegexGenerator, self).__init__(name=name, *args, **kwargs)
        