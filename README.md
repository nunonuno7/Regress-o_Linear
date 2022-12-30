# Importar o Modelo de Regressão Linear

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Defenir a variável dependete y (output) e a variavel independente x (input)
data =pd.read_csv("Salary_Data.csv")
X = data.iloc[:, 0]
Y = data.iloc[:, 1]
plt.scatter(X, Y)
plt.show()

# Importar a nossa classe e correr o modelo

from LinearRegressionGD import LinearRegressionGD
model = LinearRegressionGD(learning_Rate=0.0001,max_iter=1000,min_delta_iter=0.0002)
model.fit(X,Y)

# Razão da Paragem min_delta_iter ou max_iter?
model.reason

# plot com as observações e o y_pred

plt.scatter(X, Y) 
plt.plot([min(X), max(X)], [min(model.Y_pred), max(model.Y_pred)], color='red')  # regression line
plt.show()
# Precisão de Outputs

x_ = [4.5,8,9]
preditct = [model.predict(_) for _ in x_]
# Precisão de Outputs com um dado promt.input

model.predict_input()