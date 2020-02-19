import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D


class Perceptron:
    """Perception implementation"""

    def __init__(self, N, alpha=0.1):
        """
        Key Arguments:
        N: The number of columns in our input feature vectors
        alpha: Our learning rate for the Perceptron algorithm
        """
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                p = self.step(np.dot(x, self.W))

                if p != target:
                    error = p - target
                    self.W += -self.alpha * error * x

    def predict(self, X, add_bias=True):
        X = np.atleast_2d(X)

        if add_bias:
            X = np.c_[X, np.ones((X.shape[0]))]

        return self.step(np.dot(X, self.W))


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        """
        Key Arguments:
        layers: A list of integers which represents the actual architecture
                of the feedforward network
        alpha: Our learning rate for the Perceptron algorithm
        """
        self.W = []
        self.layers = layers
        self.alpha = alpha

        for i in np.arange(0, len(layers) - 2):
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        return "NeuralNetwork: {}".format(
            "-".join(str(l) for l in self.layers)
        )

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, display_update=100):
        X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            if epoch == 0 or (epoch + 1) % display_update == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))

    def fit_partial(self, x, y):
        """
        Backpropagation algorithm

        Key Arguments:
        x: An individual data point from our design matrix
        y: The corresponding class label
        """
        # Feedforward Phase
        A = [np.atleast_2d(x)]

        for layer in np.arange(0, len(self.W)):
            net = A[layer].dot(self.W[layer])
            out = self.sigmoid(net)
            A.append(out)

        # Backpropagation Phase
        error = A[-1] - y
        D = [error * self.sigmoid_derivative(A[-1])]

        for layer in np.arange(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_derivative(A[layer])
            D.append(delta)

        # Weight Update Phase
        D = D[::-1]

        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, add_bias=True):
        p = np.atleast_2d(X)

        if add_bias:
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p

    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, add_bias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss


class ShallowNet:

    @staticmethod
    def build(width, height, depth, classes):
        """Construct the Shallow network architecture

        Key Arguments:
        width: The width of the input images
        height: The height of our input images
        depth: The number of channels in the input image
        classes: The total number of classes that our network should learn
        """
        model = Sequential()
        input_shape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)

        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model


class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        """Construct the LeNet network architecture

        Key Arguments:
        width: The width of the input images
        height: The height of our input images
        depth: The number of channels in the input image
        classes: The total number of classes that our network should learn
        """
        model = Sequential()
        input_shape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)

        model.add(Conv2D(20, (5, 5), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model


class MiniVGGNet:

    @staticmethod
    def build(width, height, depth, classes):
        """Construct the MiniVGGNet network architecture

        Key Arguments:
        width: The width of the input images
        height: The height of our input images
        depth: The number of channels in the input image
        classes: The total number of classes that our network should learn
        """
        model = Sequential()
        input_shape = (height, width, depth)
        channel_dim = -1

        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            channel_dim = 1

        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
