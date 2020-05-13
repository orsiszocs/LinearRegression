from controller import read_file, gradient_descent, to_string, least_squares


def start():
    data = read_file()
    while True:
        try:
            method = int(input("1. gradient descent\n2. least squares\n"))
            if method != 1 and method != 2:
                raise ValueError
            if method == 1:
                learning_rate = float(input("learning rate: "))
                if learning_rate <= 0:
                    raise ValueError
                number_iterations = int(input("number iterations: "))
                if number_iterations < 1:
                    raise ValueError
                error, function = gradient_descent(data, learning_rate, number_iterations)
                print("average error is " + to_string(error))
                print("found function is " + function)
            if method == 2:
                error, function = least_squares(data)
                print("average error is " + to_string(error))
                print("found function is " + function)
        except ValueError:
            print("invalid input")


start()
