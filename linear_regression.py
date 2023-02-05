import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as pyplot
import pickle
import argparse
import json

from sklearn import linear_model
from sklearn.utils import shuffle
from matplotlib import style

def parse_data_to_pd_df(file_name):
    """This function imports the data from the csv file 

        Args:
            file_name (str): The filename

        Returns:
            data (pandas.DataFrame): The pandas Dataframe
    """
    data = pd.read_csv(file_name, sep=";")

    return data

def feature_selection(data):
    """This function performs the feature selection
    
        Args:
            data (pandas.Dataframe): Dataframe createad using the csv data imported

        Returns:
            data (pandas.Dataframe): Dataframe obtained after feature selection
    """
    data=data[["G1","G2","G3","studytime","failures","absences"]]

    return data

def preprocess_data(data, objective_variable):
    """This function preprocesses the data and prepare it to be trained.
        Removes our objective variable from the dataset  and creates dataframe X
        Includes only the objective variable and creates dataframe y
    
        Args:
            data (pandas.Dataframe): Dataframe obtained after feature selection
            objective_variable (str): Variable defined by the user to be the objective variable
        
        Returns:
            X (pandas.Dataframe): Dataframe that contains explanatory variables
            y (pandas.Dataframe): Dataframe that contains the objective variable
    """
    X=np.array(data.drop([objective_variable], 1))
    y=np.array(data[objective_variable])

    return X, y

def train_model(n_times, X, y):
    """This function trains the model multiple times (e.g. 30x) 
        through different samples between test and train set (90-10) 
        to find the best model, collecting the score and saving the model.

        Args:
            n_times (int): The number of times defined by the user that the model will be trained
            X (pandas.Dataframe): Dataframe that contains explanatory variables
            y (pandas.Dataframe): Dataframe that contains the objective variable

        Returns:   
            best_acc (float): variable that collects the best accuracy after 
                                training the model n_times
            set_used (list): list tha contains X_train, X_test, y_train, y_test
    """
    best_acc = 0
    set_used = []
    for _ in range (n_times):
        X_train, X_test, y_train, y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.1)

        #Model training
        linear=linear_model.LinearRegression()
        linear.fit(X_train, y_train)
        
        #Accuracy computation
        acc=linear.score(X_test,y_test)

        if acc > best_acc: 
            best_acc=acc
            set_used = [X_train, X_test, y_train, y_test]
            #Saving the model to a pickle file
            with open("grade_model.pickle","wb") as f:
                pickle.dump(linear,f)

    return set_used, best_acc

def load_model():
    """This function loads the model saved

        Args:

        Returns:
            linear_model (sklearn.linear_model): model loaded using the pickle file
            that contains the best model based on the iteration
    """
    pickle_in=open("grade_model.pickle","rb")

    linear_model = pickle.load(pickle_in)
    return linear_model

def predict(linear_model, X_test, y_test):
    """This function is used to predict the grade of the students included on the test sample
    
        Args:
            linear_model (sklearn.linear_model): best linear regression model trained
            X_test (pandas.Dataframe): Dataframe that includes the explanatory variables for the test set
            y_test (pandas.Dataframe): Dataframe that includes the objective variable for the test set

        Returns:
            predictions (dict): Dictionary with the Student defined number as key 
            and other dictionary as value. This inside dictionary contains the grade
            prediction, explanatory variables used and actual grade for each student
    """
    
    predicted=linear_model.predict(X_test)
    
    predictions = {}

    for i in range(len(predicted)):
        predictions.update({
            f"Student {i + 1}": {
                "Grade_Predictions": round(predicted[i]),
                "Explanatory Variables": X_test[i].tolist(),
                "Atual Grade": int(y_test[i])
            }
        })

    return predictions


def write_to_json(predictions):
    """ This function writes the the grade prediction, explanatory variables used 
        and actual grade for each student actual grade to a json file
    
        Args:
            predictions (dict)
    """
    with open("grade_predictions.json", "w") as file:
        json.dump(predictions, file)
    

#-----------------------------------------Plotting---------------------------------------------#


def parse_arguments():
    """This function parses the argument(s) of this model

        Args:
            n_times : name of the command line field to insert on the runtime
            objective_variable : name of the command line field to insert also on runtime

        Return:
            args: Stores the extracted data from the parser run
    """

    parser = argparse.ArgumentParser(description='Process all the arguments for this model')
    parser.add_argument('file_name', help='The csv file name')
    parser.add_argument('n_times', help='The number of times that the model will be trained', type=int)
    parser.add_argument('objective_variable', help="The objective variable")

    args = parser.parse_args()

    return args

def main():
    """This is the main function of this Linear Model Regression Implementation model"""
    args = parse_arguments()
    
    data = parse_data_to_pd_df(args.file_name)

    data = feature_selection(data)
    
    X, y = preprocess_data(data, args.objective_variable)

    p="studytime"
    style.use("ggplot")
    pyplot.scatter(data[p],data["G3"])
    pyplot.xlabel(p)
    pyplot.ylabel("Final Grade")
    pyplot.show()

    set_used, best_acc = train_model(args.n_times, X, y)

    linear_model = load_model()

    X_test = set_used[1]
    print(type(X_test))
    y_test = set_used[3]

    predictions = predict(linear_model, X_test, y_test)
    
    print("Evaluating the model and Interpreting...")

    #Accuracy
    print("Accuracy: \n" + str(best_acc))

    #Linear Coefficients   
    print("Linear Coefficients: \n" , linear_model.coef_)

    #Linear Intercept
    print("Intercept: \n" , linear_model.intercept_)

    print("Writing grade prediction vs actual grade to a json file...")
    write_to_json(predictions)


if __name__ == '__main__':
    main()
