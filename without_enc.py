import pandas as pd
import numpy as np
from sklearn import linear_model


def set_value(row_number, assigned_value):
    return assigned_value[row_number]

df =pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                names=["sepal length","sepal width","petal length","petal width","Class"])

event_dictionary ={'Iris-setosa' : 1, 'Iris-versicolor' : 2, 'Iris-virginica' : 3}
  
df['y'] = df['Class'].map(event_dictionary)

reg= linear_model.LinearRegression()

reg.fit(df[['sepal length','sepal width','petal length','petal width']],df.y)

w= reg.coef_
print(w)

#df=pd.read_csv("insurance.csv")
#reg= linear_model.LinearRegression()
#reg.fit(df[['age','bmi','children','smoker','region']],df['charges'])