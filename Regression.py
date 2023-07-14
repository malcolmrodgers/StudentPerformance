import pandas as pd
import numpy as np
import sklearn.model_selection
import tensorflow
from sklearn import linear_model

##load data from csv (UCI Machine Learning Repo)
data = pd.read_csv("G:\StudentPerformance\Data\student-mat.csv", sep=";")
print(data.head())

##Narrow down into useful attributes
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())

#column we want our model to predict
predict = "G3"
#separate data set into array without predict, and array with only predict
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

#split x and y into training and testing segments using scikit-learn
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

linear = linear_model.LinearRegression()

#fit sklearn linear model to x and y training splits
linear.fit(x_train, y_train)
#Use x and y test splits to get accuracy value
acc = linear.score(x_test, y_test)
print(acc)

#print coefficient and intercept
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

#make predictions based on test data
predictions = linear.predict(x_test)
#print predictions alongside test inputs
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
