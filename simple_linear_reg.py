
#import the needed packages!!
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,[0]].values
y=dataset.iloc[:,[-1]].values

#plot the values
plt.scatter(X, y, color = 'red')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Splitting the dataset into the train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)#keep random_state value same to get the same result!!

# simple linear regression model on the training set with ordinary least square
from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(X_train, y_train)

# Predicting the Test set results
y_pred1 = regressor1.predict(X_test)

#simple linear regression model on the training set with stochastic gradient descent
from sklearn import linear_model
regressor2 = linear_model.SGDRegressor(max_iter=1000, tol=1e-3,random_state=0)
regressor2.fit(X_train, y_train)

#prediciting the tesy set results
y_pred2= regressor2.predict(X_test)


# Visualising the Training set results with both the methods
#blue line is ols and pink line is sgd
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor1.predict(X_train), color = 'blue')
plt.plot(X_train, regressor2.predict(X_train), color = 'pink')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results with both the methods
#blue line is ols and pink line is sgd
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor1.predict(X_train), color = 'blue')
plt.plot(X_train, regressor2.predict(X_train), color = 'pink')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#r2_score for both the methods
from sklearn.metrics import r2_score
#r2 for OLS model
r_squared1 = r2_score(y_test, y_pred1)

#r2 for SGD model
r_squared2 = r2_score(y_test, y_pred2)
#print(r_squared1,r_squared2)


