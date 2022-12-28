# Regress-o_Linear
# Making the imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 9.0)

# Preprocessing Input data
path_to_file = r"C:\Users\nunov\Regressão Linear\Regress-o_Linear\Salary_Data.csv"
data =pd.read_csv(path_to_file)
X = data.iloc[:, 0]
Y = data.iloc[:, 1]
plt.scatter(X, Y)
plt.show()
def mse(Y_pred, Y):
  return np.mean((Y_pred - Y)**2)

# Building the model
m = 0
b = 0

learning_Rate = 0.0001  # The learning Rate
max_iter = 1000  # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

# Performing Gradient Descent  
for i in range(max_iter): 
    Y_pred = m*X + c  # The current predicted value of Y
    error = Y - Y_pred
    D_m = (-2/n) * sum(X * (error))  # Derivative wrt m
    D_b = (-2/n) * sum(error)  # Derivative wrt b
    m = m  - learning_Rate * D_m  # Update m
    b = b  - learning_Rate * D_b  # Update b
print (m, b)
# Making predictions
Y_pred = m*X + b

plt.scatter(X, Y) 
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
plt.show()
def previsão():
    a= int(input('Qual é o Valor de X?'))
    Y_pred = m*a + b
    print(f' Para um X de {a} o Output esperado é {Y_pred}')

previsão()