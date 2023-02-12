import numpy as np
import sklearn
import argparse

from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics


def pre_processing():
    """"This function loads and preprocesses the data
    
        Args:
            
        Returns:
            x(numpy.array): Array that contains all features
            y(numpy.array): Array that contains the labels
    """
    digits=load_digits()
    x=scale(digits.data) #scale all our features, because our digits will have large values. -> To save time in the computation
    y=digits.target  

    return x,y


def bench_k_means(estimator, name, x,y):
    """This function trains the model collecting the accuracy

        Args:
            x(numpy.array): Array that contains all features
            y(numpy.array): Array that contains the labels
            
        Returns:   
            
    """
    #Estimator Fit
    estimator.fit(x)
    print(f"inertia \t{str(round(estimator.inertia_))}",
    f"\nhomogeneity_score\t{str(round(metrics.homogeneity_score(y, estimator.labels_), 3))}",
    f"\ncompleteness_score\t{str(round(metrics.completeness_score(y, estimator.labels_), 3))}",
    f"\nv_measure_score \t{str(round(metrics.v_measure_score(y, estimator.labels_), 3))}",
    f"\nadjusted_rand_score \t{str(round(metrics.adjusted_rand_score(y, estimator.labels_), 3))}",
    f"\nadjusted_mutual_info_score \t{str(round(metrics.adjusted_mutual_info_score(y, estimator.labels_), 3))}",
    f"\nsilhouette_score \t{str(round(metrics.silhouette_score(x, estimator.labels_, metric='euclidean'), 3))}"    )
    
    return 


def parse_arguments():
    """This function parses the argument(s) of this model

        Args:
            k: name of the command line field to insert on the runtime
            init: name of the command line field to insert on the runtime
            n_init: name of the command line field to insert on the runtime

            
          
        Return:
            args: Stores the extracted data from the parser run
   """

    parser = argparse.ArgumentParser(description='Process all the arguments for this model')
    parser.add_argument('k', help= 'Specifies the number of clusters to form as well as the number of centroids to generate (int)', type=int)
    parser.add_argument('init', help= 'Specifies the method for initialization: random, k-means++ (str)')
    parser.add_argument('n_init', help= 'Specifies the number of times the k-means algorithm is run with different centroid seeds (int)', type=int)

    args = parser.parse_args()

    return args


def main():
    """This is the main function of this K-means Implementation model
    """
    args = parse_arguments()
    x, y= pre_processing()
    clf = KMeans(n_clusters=args.k, init=args.init, n_init=args.n_init)

    x=bench_k_means(clf, "1", x,y)
      


if __name__ == '__main__':
    main()
