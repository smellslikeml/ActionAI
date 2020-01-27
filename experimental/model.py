import config as cfg
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical

def lstm_model():
    model = Sequential()
    model.add(LSTM(16, dropout=0.2, recurrent_dropout=0.2, input_shape=(cfg.pose_vec_dim, cfg.window)))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(cfg.activity_dict), activation='softmax'))
    print(model.summary())
    return model

