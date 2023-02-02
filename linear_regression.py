import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

#Import Data
data = pd.read_csv("student-mat.csv",sep=";")

#Feature Selection
data=data[["G1","G2","G3","studytime","failures","absences"]]

#Objective Variable 
predict = "G3"

#Removing our objective variable from the dataset
x=np.array(data.drop([predict],1))
y=np.array(data[predict])

x_train, y_train, x_text, y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.1)