import sklearn
import argparse

from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


def pre_processing():
    """"This function loads and preprocesses the data
    
        Args:
            
        Returns:
            pre_processing_data (tupple): list that contains the explanatory variables
            and objective variable
    """
    cancer=datasets.load_breast_cancer()
    x=cancer.data
    y=cancer.target
    
    pre_processing_data=[x,y]

    return pre_processing_data


def split_train_test(pre_processing_data):
    """ This functions splits the dataset between test and train

        Args:
            pre_processing_data (tupple): list that contains the explanatory variables
            and objective variable
            
        Returns:
            set_used (list): list that contains x_train, x_test, y_train, y_test

    
    """
    x=list(pre_processing_data[0])
    y=list(pre_processing_data[1])

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    set_used=[x_train, x_test, y_train, y_test]

    return set_used


def train_model(set_used,kernel, C):
    """This function trains the model collecting the accuracy

        Args:
            set_used (list): list tha contains x_train, x_test, y_train, y_test
            
        Returns:   
            acc (float): variable that collects the accuracy 
    """
    #Defining classes
    classes=['malignant' , 'benign']
    
    #Model training
    clf=svm.SVC(kernel=kernel, C=C)
    clf.fit(set_used[0],set_used[2])
    
    predicted = clf.predict(set_used[1])

    #Accuracy computation
    acc=metrics.accuracy_score(set_used[3], predicted)

    return acc


def parse_arguments():
    """This function parses the argument(s) of this model

        Args:
            kernel: name of the command line field to insert on the runtime
            C: name of the command line field to insert on the runtime
          
        Return:
            args: Stores the extracted data from the parser run
   """

    parser = argparse.ArgumentParser(description='Process all the arguments for this model')
    parser.add_argument('kernel', help= 'Specifies the kernel type to be used in the algorithm: linear, poly, rbf, sigmoid, precomputed')
    parser.add_argument('C', help= 'Regularization parameter',type=int)
    args = parser.parse_args()

    return args


def main():
    """This is the main function of this SVM Implementation model
    """
    args = parse_arguments()

    pre_processing_data=pre_processing()
    
    set_used=split_train_test(pre_processing_data)
    
    acc=train_model(set_used, args.kernel, args.C)
    
    print("Evaluating the model...")

    print("Accuracy: \n" + str(acc))



if __name__ == '__main__':
    main()

