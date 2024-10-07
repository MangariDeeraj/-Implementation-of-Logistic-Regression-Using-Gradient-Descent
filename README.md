# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 1. Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn..  

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:MANGARI DEERAJ 
RegisterNumber: 212223100031 
*/


dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataimport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('Placement_Data.csv')
dataset
dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')set["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
Y
theta = np.random.randn(X.shape[1])
y = Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta

theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
```

## Output:


Dataset

![image](https://github.com/preethi2831/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155142246/5bfd1bab-c917-439a-b704-ccc834b3c6bb)


Data types

![image](https://github.com/preethi2831/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155142246/673238d6-78f7-4f11-ae12-7a4d37373d34)

New dataset

![image](https://github.com/preethi2831/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155142246/2f3befdc-e483-4622-b528-d094fd91bcf4)

Y values

![image](https://github.com/preethi2831/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155142246/62bcf3a1-a6d6-4105-852d-31b05f5cab2d)

Accuracy

![image](https://github.com/preethi2831/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155142246/4b336b75-ee0a-488d-898a-ec7c66288674)

Y pred

![image](https://github.com/preethi2831/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155142246/ede9b220-c365-4f22-914c-5d0b62c1c720)

New Y

![image](https://github.com/preethi2831/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155142246/ea7536cc-4ec7-443b-8c69-82a311110908)

![image](https://github.com/preethi2831/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155142246/a1e09cc3-458a-45a3-ba60-deaf87e554e8)

![image](https://github.com/preethi2831/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/155142246/e5795b77-f293-4244-b6aa-40a4414635cc)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

