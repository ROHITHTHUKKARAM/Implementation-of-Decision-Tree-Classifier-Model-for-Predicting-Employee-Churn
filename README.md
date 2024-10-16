![image](https://github.com/user-attachments/assets/a44937fc-e979-4cc0-8a19-09ba3d665e3f)# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1:Import the required libraries.

Step 2:Upload and read the dataset.

Step 3:Check for any null values using the isnull() function.

Step 4:From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

Step 5:Find the accuracy of the model and predict the required values by importing the required module from sklearn.
 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Rohith T
RegisterNumber: 212223040173
*/
import pandas as pd
data = pd.read_csv("/Users/admin/Desktop/ML/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company",
          "Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
![Screenshot 2024-10-16 221453](https://github.com/user-attachments/assets/3b011bed-9b37-4228-b2bd-da5c570bae19)
![Screenshot 2024-10-16 221527](https://github.com/user-attachments/assets/10f21b0b-5f24-4967-80ff-fab3e78e9025)





## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
