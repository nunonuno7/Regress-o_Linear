
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegressionGD:

    """Simple Linear regression model using gradient descent.

    Parameters
    ----------
    learning_Rate : float, optional
        The learning rate (between 0.0 and 1.0).
    max_iter : int, optional
        The number of maximum training iterations.
    min_delta_iter: float, optional
        The minimal change between delta iterations (between 0.0 and 1.0).

    """

    def __init__(self,learning_Rate=0.001,max_iter=10000,min_delta_iter=0.0001):
       self.learning_Rate = learning_Rate
       self.max_iter = max_iter
       self.min_delta_iter = min_delta_iter

    def fit(self,X,Y):
        """Fit the model to the training data.

        Parameters
        ----------
        X : array-like, shape (n_samples)
            The training data.
        y : array-like, shape (n_samples,)
            The target values.

        Returns
        -------
        Values of "m" and "b" where y = m*X + b

        """

        global new_error
        global old_error
        global delta_iter
        self.m = 0
        self.b = 0
        n = float(len(X))

        for i in range(self.max_iter+1):
            self.Y_pred = self.m*X + self.b  # The current predicted value of Y
            dif = Y - self.Y_pred
            D_m = (-2/n) * sum(X * (dif))  # Derivative wrt m
            D_b = (-2/n) * sum(dif)  # Derivative wrt b
            old_error = np.sum(((X*self.m+self.b)-Y)**2)
            self.m = self.m  - self.learning_Rate * D_m  # Update m
            self.b = self.b  - self.learning_Rate * D_b  # Update b
            new_error = np.sum(((X*self.m+self.b)-Y)**2)
            delta_iter = ((new_error-old_error)/old_error)
            if abs(delta_iter) < self.min_delta_iter:
                self.reason = (f'Parou com um delta_iter de {round(delta_iter,5)} ')
                break
            elif i == self.max_iter:
                self.reason = (f'Parou com o max_iter de {self.max_iter}')

        return self.m, self.b

    def predict(self,X):
        """Predict the output for a given input X."""
        return print(f'Para um X de {X} o Output esperado é {round((X*self.m +self.b),2)}')
        

    def predict_input(self):
        
        """This function predicts the output of a given X provided by a prompt(input) of X.

        The input only accepts numbers, but there is a possibility to exit the loop with a word in the list:

        ['quit','sair','leave', 'quit','exit','parar','stop',' ']"""

        flag = True
        while flag:
            a= input('Qual é o Valor de X?')
            if a in ['quit','sair','leave', 'quit','exit','parar','stop',' ']:
                flag = False
            else:
                try:
                    a= float(a)
                    Y_pred_ = self.m*a + self.b
                    print(f'Para um X de {a} o Output esperado é {round(Y_pred_,2)}')
                    flag = False
                except:
                    n = print('Deve ser selecionado um número, ou saia da função, por favor, tente mais uma vez!')