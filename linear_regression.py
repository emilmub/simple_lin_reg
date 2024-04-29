import numpy as np

class LinearRegression:

    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.residual_sum = None

    def fit(self):
        numerator = 0
        self.denominator = 0
        self.x_mean = np.mean(self.x_data)
        self.y_mean = np.mean(self.y_data)
        for x,y in zip(self.x_data, self.y_data):
            x_dif = x - self.x_mean
            y_dif = y - self.y_mean
            numerator += x_dif*y_dif
            self.denominator += x_dif*x_dif
        self.beta_1 = numerator/self.denominator
        self.beta_0 = self.y_mean - self.beta_1*self.x_mean
        return self.beta_0, self.beta_1

    def RSS(self):
        self.residual_sum = 0
        for x,y in zip(self.x_data, self.y_data):
            residual = y - self.beta_0 - self.beta_1*x
            self.residual_sum += residual*residual
        return self.residual_sum

    def RSE(self):
        self.sigma = np.sqrt(self.RSS()/(len(self.x_data)-2))
        return self.sigma

    def std_errors(self):
        n = len(y_data)
        self.RSE()
        self.SE_beta_0 = self.sigma*(1/n + self.x_mean*self.x_mean/self.denominator)
        self.SE_beta_1 = self.sigma*self.sigma/self.denominator
        return self.SE_beta_0, self.SE_beta_1

    def R_squared(self):
        self.TSS = 0
        for y in self.y_data:
            self.TSS += (y - self.y_mean)*(y - self.y_mean)
        self.R2 = 1 - self.RSS()/self.TSS
        return self.R2