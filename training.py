from vae2 import VAE
from database_management import Database
import ast
import numpy as np
from keras.utils import pad_sequences

db = Database()
dataset = db.get_table()
dataset = [i[1:] for i in dataset]
ast.