import sklearn
import pandas as pd
import numpy as np
import argparse
import json

from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier


def parse_data_to_pd_df(file_name):
    """This function imports the data from the csv file 

        Args:
            file_name (str): The filename

        Returns:
            data (pandas.DataFrame): The pandas Dataframe
   """
    data = pd.read_csv(file_name)

    return data


def pre_processing(data):
    """"This function preprocesses the data
    
        Args:
            data (pandas.Dataframe): The pandas Dataframe

        Returns:
            pre_processing_data (tupple): list that contains the explanatory variables
            and objective variable
    """

    le = preprocessing.LabelEncoder()
    buying = le.fit_transform(list(data["buying"]))
    maint = le.fit_transform(list(data["maint"]))
    door = le.fit_transform(list(data["door"]))
    persons = le.fit_transform(list(data["persons"]))
    lug_boot = le.fit_transform(list(data["lug_boot"]))
    safety = le.fit_transform(list(data["safety"]))
    cls = le.fit_transform(list(data["class"]))
    
    pre_processing_data=[buying,maint,door,persons,lug_boot,safety,cls]

    return pre_processing_data
    

def split_train_test(pre_processing_data):
    """ This functions splits the dataset between test and train

        Args:
            pre_processing_data (tupple): list that contains the explanatory variables
            and objective variable
            
        Returns:
            set_used (list): list tha contains x_train, x_test, y_train, y_test  
    """
    x=list(zip(pre_processing_data[0],pre_processing_data[1],pre_processing_data[2]+pre_processing_data[3]+pre_processing_data[4],pre_processing_data[5]))
    y=list(pre_processing_data[6])

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    set_used=[x_train, x_test, y_train, y_test]

    return set_used


def train_model(set_used,n_neighbors):
    """This function trains the model collecting the accuracy

        Args:
            set_used (list): list tha contains x_train, x_test, y_train, y_test
            n_neighbors (int): The number of neighbors defined by the user
        Returns:   
            acc (float): variable that collects the accuracy 
    """
    #Model training
    model=KNeighborsClassifier(n_neighbors)
    model.fit(set_used[0],set_used[2])
    
    #Accuracy computation
    acc=model.score(set_used[1],set_used[3])

    predicted = model.predict(set_used[1])
    names = ["unacc", "acc", "good", "vgood"]
    for i in range(len(predicted)):
        print("Predicted: ", names[predicted[i]], "Data: ", set_used[1][i], "Actual: ", names[set_used[3][i]])

    return acc


def parse_arguments():
    """This function parses the argument(s) of this model

        Args:
            file_name: name of the command line field to insert on the runtime
            n_neighbors: name of the command line field to insert on the runtime
        Return:
            args: Stores the extracted data from the parser run
   """

    parser = argparse.ArgumentParser(description='Process all the arguments for this model')
    parser.add_argument('file_name', help='The csv file name')
    parser.add_argument('n_neighbors', help='The number of neighbors',type=int)
   
    args = parser.parse_args()

    return args


def main():
    """This is the main function of this KNN Implementation model
    """
    args = parse_arguments()
    
    data = parse_data_to_pd_df(args.file_name)

    pre_process_data=pre_processing(data)

    set_used=split_train_test(pre_process_data)
   
    x_test = set_used[1]

    y_test = set_used[3]

    model, acc = train_model(set_used,args.n_neighbors)

    print("Evaluating the model...")

    print("Accuracy: \n" + str(acc))


if __name__ == '__main__':
    main()
