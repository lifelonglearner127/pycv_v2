import numpy as np


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
