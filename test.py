import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

def main(n = 100,b0 = 0, b1 = 1):
    x_data = np.random.rand(n,1)*100
    y_data = b0 + b1*x_data
    y_data += np.random.normal(scale = 80,size = (n,1))
    
    LR = LinearRegression(x_data,y_data)
    LR.fit()
    
    y_hat = LR.beta_0 + LR.beta_1*x_data

    print("RSE:", LR.RSE())
    print("R squared:", LR.R_squared())
    

    plt.scatter(x_data,y_data)
    plt.plot(x_data,y_hat, 'r')
    plt.show()

if __name__ == "__main__":
    n = 100
    b0 = 30
    b1 = 5
    main(n,b0,b1)