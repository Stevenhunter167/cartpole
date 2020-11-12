import numpy as np
import tensorflow as tf


import builtins
def print(*args, **kwargs):
    builtins.print(*args, **kwargs)


class Learner:
    def __init__(self):
        # policy network 4-128-128-128-128-2
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_shape=(4,), activation='relu'),
            tf.keras.layers.Dropout(0.8),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.8),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.8),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.8),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.8),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3, name='Adam')
        self.model.compile(optimizer=opt, loss=loss_fn, metrics='acc')

    def infer(self, state) -> int:
        """ predict an action given a state """
        return np.argmax(self.model.predict(np.array([state]))[0])

    def train(self, DATA, epochs=5, verbose=0):
        np.random.shuffle(DATA)
        X, Y = DATA[:, :-1], DATA[:, -1]
        self.model.fit(X, Y, epochs=epochs, batch_size=50000, verbose=verbose)
        self.save()


    def save(self):
        pass

    def load(self):
        pass