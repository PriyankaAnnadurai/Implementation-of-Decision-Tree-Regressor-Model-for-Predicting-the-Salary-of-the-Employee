# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries .
2. Read the data frame using pandas.
3. Get the information regarding the null values present in the dataframe.
4. Apply label encoder to the non-numerical column inoreder to convert into numerical values.
5. Determine training and test data set.
6. Apply decision tree regression on to the dataframe.
7. Get the values of Mean square error, r2 and data prediction.

   

## Program:
```py
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: PRIYANKA.A
RegisterNumber:  212222230113
*/


import pandas as pd
data=pd.read_csv("/content/Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

```

## Output:

### data.head()
![276262885-d6b4cb40-1795-4b64-8683-7acc5b701051](https://github.com/PriyankaAnnadurai/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118351569/82edb450-2064-4a60-891d-8f06b67851b0)


### data.info()
![276262851-d410937e-837d-4ce0-ae0d-d18246cbb07b](https://github.com/PriyankaAnnadurai/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118351569/5901fb3f-65aa-45e2-a36a-86ff0a7d2a0e)


### isnull() & sum() function
![276262814-1ac338d8-7d40-403a-90e4-48a7d6fc03cb](https://github.com/PriyankaAnnadurai/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118351569/4c1bccec-1ad4-4588-bf62-1529fd4ba7c4)


### data.head() for position
![276262767-8b81fa42-a87e-49a8-8129-884c0e6de355](https://github.com/PriyankaAnnadurai/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118351569/4c40d996-1546-404d-b991-5647c598a46a)


### MSE value
![276262714-7afc264d-6718-4b8e-a234-aea823b90e0e](https://github.com/PriyankaAnnadurai/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118351569/0a77d690-4d22-498e-8c16-cc5371edbf03)


### R2 value
![276262661-f0e9b89b-40d0-442c-bead-4ecbfe53aa62](https://github.com/PriyankaAnnadurai/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118351569/2d04b400-7383-41f9-b93c-5d6d181cac99)


### Prediction value
![276262449-49a799ab-6bf5-45d7-8024-83e39b739903](https://github.com/PriyankaAnnadurai/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118351569/09424864-4c4c-42a2-993d-75eb319c7625)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
