import numpy as np


def initialize_parameters(size):
    w = np.zeros((size, 1))
    b = 0
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]
    A = predict(w, b, X)
    da = A - Y
    dw = 1 / m * np.dot(X, da.T)
    db = 1 / m * np.sum(da)
    # cost=1/(2*m)*np.sum(da*da)
    return dw, db


def gradient_descent(w, b, X, Y, number_iterations, learning_rate):
    for _ in range(number_iterations):
        dw, db = propagate(w, b, X, Y)
        w = w - learning_rate * dw
        b = b - learning_rate * db
    return w, b, dw, db


def predict(w, b, X):
    return np.dot(w.T, X) + b


def model_gradient_descent(X, Y, number_iterations, learning_rate):
    w, b = initialize_parameters(X.shape[0])
    w, b, dw, db = gradient_descent(w, b, X, Y, number_iterations, learning_rate)
    predictions = predict(w, b, X)
    error_matrix = predictions - Y
    error = np.average(np.multiply(error_matrix, error_matrix))
    return error, w, b


def model_least_squares(X, Y):
    w = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), Y)
    error_matrix = np.dot(X, w) - Y
    error = np.average(np.multiply(error_matrix, error_matrix))
    b = w[0, 0]
    w = np.delete(w, (0), axis=0)
    return error, w, b
