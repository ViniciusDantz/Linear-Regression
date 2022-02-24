import matplotlib.pyplot as plt
import pandas as pd
import random as rd


class LinearRegressor:
    def __init__(self) -> None:
        self.slope = 0
        self.intercept = 0

    def calculate_slope(self, X, Y):
        x_mean = X.mean()
        y_mean = Y.mean()

        product_sum = 0
        square_sum = 0

        for x, y in zip(X, Y):
            x_res = x - x_mean
            y_res = y - y_mean

            product_sum += x_res * y_res
            square_sum += x_res ** 2
        
        self.slope = product_sum / square_sum

    def calculate_intercept(self, X, Y):
        self.intercept = Y.mean() - (self.slope * X.mean())

    def calculate_residue(self, ground_truth, prediction):
        return ground_truth - prediction

    def residual_sum_squares(self, ground_truths, predictions):
        sum = 0

        for ground_truth, prediction in zip(ground_truths, predictions):
            sum += self.calculate_residue(ground_truth, prediction) ** 2

        return sum

    def predict(self, x):
        return (self.slope * x) + self.intercept

    def fit(self, X, Y):
        self.calculate_slope(X, Y)
        self.calculate_intercept(X, Y)

    def plot(self, X, Y, predictions):
        plt.scatter(X, Y, color='b')
        plt.plot(X, predictions, color='r')
        plt.show()


if __name__ == '__main__':
    m = rd.uniform(-10, 10)
    c = rd.uniform(-50, 50)

    X = pd.Series([rd.uniform(-5, 5) for i in range(100)])
    Y = pd.Series(X.apply(lambda x: m*x + c + rd.randint(-50, 50)))

    # plt.scatter(X, Y, color='b')
    # plt.show()

    linear_model = LinearRegressor()

    linear_model.fit(X, Y)

    predictions = [linear_model.predict(x) for x in X]

    print(f'Pred Line Equation: Y = {linear_model.intercept} + {linear_model.slope} * X')
    print(f'Real Line Equation: Y = {c} + {m} * X')

    print('Sum Squared Error:', linear_model.residual_sum_squares(Y, predictions))
    linear_model.plot(X, Y, predictions)