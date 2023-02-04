import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as pyplot
import pickle
from sklearn import linear_model
from sklearn.utils import shuffle
from matplotlib import style


#--------------------------------------Import and Preparing Data-------------------------------#
#Import Data
data = pd.read_csv("student-mat.csv",sep=";")

#Feature Selection
data=data[["G1","G2","G3","studytime","failures","absences"]]

#Objective Variable 
predict = "G3"

#Removing our objective variable from the dataset
X=np.array(data.drop([predict], 1))
y=np.array(data[predict])


#------------------------------------------Training-------------------------------------------#
X_train, X_test, y_train, y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.1)

#variable that collects the accuracy of each model
best=0 

#split dataset into train and test (90-10)
"""
Train the model multiple times (e.g. 30x) 
to obtain the best score
"""
for _ in range (30):
    X_train, X_test, y_train, y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.1)

    #Training the model
    linear=linear_model.LinearRegression()
    linear.fit(X_train, y_train)
    #Determine the accuracy
    acc=linear.score(X_test,y_test)


#--------------------------Saving the best model(to avoid re-training)-------------------------#
"""
Saves the model that we are training 
if the accuracy is better than any previous
accuracy that we have seen
"""
if acc > best: 
    best=acc
    with open("grademodel.pickle","wb") as f:
        pickle.dump(linear,f)

pickle_in=open("grademodel.pickle","rb")

#Load the model
linear=pickle.load(pickle_in)

#------------------------------Evaluating the model and Interpreting---------------------------#
#Accuracy
print("Accuracy: \n" + str(acc))

#Linear Coefficients
print("Co: \n" , linear.coef_)

#Linear Intercept
print("Intercept: \n" , linear.intercept_)

#-----------------------------------------Predicting-------------------------------------------#
#Use to predict a grade of a student
predicted=linear.predict(X_test)

for X in range(len(predicted)):
    print(round(predicted[X]),(X_test[X]),y_test[X])

#-----------------------------------------Plotting---------------------------------------------#
p="studytime"
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()

#---------------------------------------Main Function-----------------------------------------#
"""
parser = argparse.ArgumentParser(description='Process all the arguments for this model')
parser.add_argument('n_times', help='The item to search on Amazon Search Bar')
args = parser.parse_args()
return args
"""