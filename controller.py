import numpy as np

from regression import model_gradient_descent, model_least_squares


def read_file():
    data = []
    file = open("bdate2.txt", "r")
    for _ in range(3):
        file.readline()
    for line in file:
        if len(line) > 5:
            strings = line.strip().split()
            floats = []
            for string in strings:
                floats.append(float(string))
            data.append(floats)
    return data


def to_string(number):
    return "{:.3f}".format(number)


def make_function(w, b):
    function = "f(a, b, c, d, e) = "
    function += to_string(w[0, 0]) + " * a + "
    function += to_string(w[1, 0]) + " * b + "
    function += to_string(w[2, 0]) + " * c + "
    function += to_string(w[3, 0]) + " * d + "
    function += to_string(w[4, 0]) + " * e + " + to_string(b)
    return function


def gradient_descent(data, learning_rate, number_iterations):
    X, Y = separate_data_gradient_descent(data)
    error, w, b = model_gradient_descent(X, Y, number_iterations, learning_rate)
    function = make_function(w, b)
    return error, function


def least_squares(data):
    X, Y = separate_data_least_squares(data)
    error, w, b = model_least_squares(X, Y)
    function = make_function(w, b)
    return error, function


def separate_data_gradient_descent(data):
    input = []
    output = []
    for element in data:
        input.append(element[:5])
        output.append(element[5])
    X = np.array(input).T
    Y = np.array(output).reshape(1, len(data))
    return X, Y

def separate_data_least_squares(data):
    input = []
    output = []
    for element in data:
        input.append([1] + element[:5])
        output.append(element[5])
    X = np.array(input)
    Y = np.array(output).reshape(len(data), 1)
    return X, Y
