import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, InputLayer
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential, load_model


class NoisyAnd(Layer):
    """Custom NoisyAND layer from the Deep MIL paper"""

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(NoisyAnd, self).__init__(**kwargs)

    def build(self, input_shape):
        self.a = 10  # fixed, controls the slope of the activation
        self.b = self.add_weight(
            name="b",
            shape=(1, input_shape[3]),
            initializer="uniform",
            trainable=True,
        )
        super(NoisyAnd, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        mean = tf.reduce_mean(x, axis=[1, 2])
        res = (
            tf.nn.sigmoid(self.a * (mean - self.b)) - tf.nn.sigmoid(-self.a * self.b)
        ) / (tf.nn.sigmoid(self.a * (1 - self.b)) - tf.nn.sigmoid(-self.a * self.b))
        return res

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[3]


class PoolingSigmoid(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(PoolingSigmoid, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[1], self.output_dim),
                        trainable=True,
        )
        self.bias = self.add_weight(
            shape=(self.output_dim, ), trainable=True
        )
        super(PoolingSigmoid, self).build(input_shape)

    def call(self, x):
        x = K.sum(x, axis=0, keepdims=True)
        x = K.dot(x, self.kernel)
        x = K.bias_add(x, self.bias)
        return K.sigmoid(x)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)


def build_model(input_shape, n_classes=2):
    """Define Deep FCN for MIL, layer-by-layer from original paper"""
    model = Sequential()
    model.add(InputLayer(input_shape=(input_shape)))
    model.add(Conv2D(32, kernel_size=(2, 2), activation="relu"))
    model.add(Conv2D(64, (2, 2), activation="relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(64, (2, 2), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(128, (2, 2), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (2, 2), activation="relu"))
    model.add(Conv2D(1000, (2, 2), activation="relu"))
    model.add(Conv2D(n_classes, (1, 1), activation="relu"))
    model.add(NoisyAnd(n_classes))
    model.add(Dense(n_classes, activation="softmax"))
    model.add(PoolingSigmoid(output_dim=1))
    return model


# def train(epochs, seed, batch_size, dataset):
#     """Train FCN"""
#     np.random.seed(seed)

#     model = define_model(dataset.input_shape, dataset.num_classes)
#     model.compile(loss=tf.keras.losses.categorical_crossentropy,
#                   optimizer=tf.keras.optimizers.Adadelta(),
#                   metrics=['accuracy'])
#     model.fit(dataset.x_train, dataset.y_train,
#               batch_size=batch_size,
#               epochs=epochs,
#               verbose=1,
#               validation_data=(dataset.x_test, dataset.y_test))
#     return model
