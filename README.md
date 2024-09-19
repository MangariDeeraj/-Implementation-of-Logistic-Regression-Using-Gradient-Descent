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
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
data =np.loadtxt("/content/ex2data1(2).txt",delimiter=",")
x=data[:,[0,1]]
y=data[:,2]
x[:5]
y[:5]
plt.figure()
plt.scatter(x[y ==1][:,0],x[y==1][:,1],label="Admitted")
plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not admitted")
plt.xlabel("exam 1 score")
plt.ylabel("exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))
plt.plot()
x_plot=np.linspace(-10,10,100)
plt.plot(x_plot,sigmoid(x_plot))
plt.show()

def costfunction(theta,x,y):
  h=sigmoid(np.dot(x,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad
x_train = np.hstack((np.ones((x.shape[0],1)),x))
theta= np.array([0,0,0])
j,grad=costfunction(theta,x_train,y)
print(j)
print(grad)
x_train = np.hstack((np.ones((x.shape[0],1)),x))
theta= np.array([-24,0.2,0.2])
j,grad=costfunction(theta,x_train,y)
print(j)
print(grad)
def cost(theta,x,y):
  h=sigmoid(np.dot(x,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]

  return j
def gradient(theta,x,y):
   h=sigmoid(np.dot(x,theta))
  
   grad=np.dot(x.T,h-y)/x.shape[0]
   return grad  
   
x_train = np.hstack((np.ones((x.shape[0],1)),x))
theta= np.array([0,0,0])
res=optimize.minimize(fun=cost, x0=theta , args=(x_train , y), method ="Newton-CG", jac=gradient)

print(res.fun)
print(res.x)
  
def plotDecisionBoundry(theta,x,y):
  x_min,x_max = x[:,0].min()-1,x[:,0].max()+1
  y_min,y_max = x[:,1].min()-1,x[:,1].max()+1
  xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  x_plot = np.c_[xx.ravel(),yy.ravel()]
  x_plot = np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot = np.dot(x_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(x[y == 1][:,0],x[y == 1][:,1],label='Admitted')
  plt.scatter(x[y == 0][:,0],x[y == 0][:,1],label='Not Admitted')
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel('Exam  1 score')
  plt.ylabel('Exam 2 score')
  plt.legend()
  plt.show()
  
prob =sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,x):
  x_train=np.hstack((np.ones((x.shape[0], 1)) , x))
  prob =sigmoid(np.dot(x_train,theta))
  return (prob>=0.5).astype(int)
np.mean(predict(res.x,x)==y)  

```

## Output:
1.Array Value of x

![image](https://github.com/user-attachments/assets/39b9ca5a-b2cf-4c91-baa7-d929b94dd275)

2.Array Value of y

![image](https://github.com/user-attachments/assets/993c090e-1a7f-4428-af3b-156fcbe081b0)

3.Exam 1 - score graph

![image](https://github.com/user-attachments/assets/64d4eea2-e799-49cb-930c-af59b42cb42d)

4.Sigmoid function graph

![image](https://github.com/user-attachments/assets/90b14f65-05b0-4e2e-884b-5b56ca1d9952)

5.X_train_grad value

![366715464-a8485e09-ac03-44c9-9abe-201cd75d12c9](https://github.com/user-attachments/assets/c8a6a24b-6f63-4981-b41e-72fc677ae3d8)

6.Y_train_grad value

![366715584-8b49d51b-8e08-4f31-880f-3c9ffd8c1ff6](https://github.com/user-attachments/assets/7030cb24-32f9-41bd-b4d1-c7e2f73805b0)

7.Print res.x

![366715617-8d2fbc79-d198-4ad3-8e3e-5db802c4b852](https://github.com/user-attachments/assets/21a9eb93-e53c-40b6-8151-33e5848cb5b6)

8.Decision boundary - graph for exam score

![366715648-6db245a0-9872-45ac-b1c6-c626a4b3127d](https://github.com/user-attachments/assets/9da2015d-d128-4e69-97e3-53990baade0d)

9.Proability value

![366715667-49b252da-2469-48af-b6da-da295af8fd89](https://github.com/user-attachments/assets/68979a63-9a8d-437c-b254-6c4cd05f84d1)

10.Prediction value of mean

![366715757-62869579-ab78-4b05-95ba-c4f09f0a84ac](https://github.com/user-attachments/assets/a51bd6f6-df92-44c3-9bbe-ff26f45273be)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

