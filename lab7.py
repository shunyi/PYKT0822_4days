import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

data1 = datasets.make_regression(100, 1, noise=1)
print(type(data1))
print(data1[0].shape, data1[1].shape)
print(np.array([1, 3, 5, 7]).shape)
print(np.array([[1], [3], [5], [6]]).shape)
plt.scatter(data1[0], data1[1], c='red', marker='^')
plt.show()
regression1 = linear_model.LinearRegression()
regression1.fit(data1[0], data1[1])
print(f'coef ={regression1.coef_}, intercept={regression1.intercept_}')
print(f"regression score = {regression1.score(data1[0], data1[1])}")
range1 = [-3, 3]
plt.plot(range1, regression1.coef_ * range1 + regression1.intercept_, c='blue')
plt.scatter(data1[0], data1[1],c='red', marker='^')
plt.show()
