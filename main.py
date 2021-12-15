################################################################################
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import argparse
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
#from sklearn.externals.six.moves import xrange
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import os
import glob
import sys
import logging
import time

################################################################################
def get_args():
    """ This function serves as a commandline bridge to take arguments that can directly modify the
    model construction. Only 2 such parameters are provided which are @criterion: the criteria
    function for Random Forest model and @data_path: the path to the dataset"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--criterion", type = str, default = "gini", help ="criterion for random\
                                                                           forest")
    u_name=  os.getlogin()
    parser.add_argument("--data_path", type = str, default = rf"C:/Users/{u_name}/Documents/anomaly_dataset"\
        , help= "path to dataset folder, make sure there are only csv files with data in it")
    parser.add_argument("--join", action= "store_true", default = False, help ="flag to join all the data files ")
    parser.add_argument("--search_best", action = "store_true", default = False, help ="perform best model search")

    return parser.parse_args()

################################################################################

def process_data(data):
    """This function performs the pre-processing necessary to input to the classifier.
    The input is the @data: the raw csv file data which can be a numpy array or a pandas dataframe.
    The output is the processed data @x: the input features and @y: the anomaly class.
    By default this function assumes the last column of csv file is the anomaly class and the rest are the features"""
    if isinstance(data, np.ndarray):
        x= data[:,:-1]
        y= data[:,-1]
        return x,y
    elif isinstance(data,pd.core.frame.DataFrame):
        data = data.to_numpy()
        return process_data(data)
    else:
        assert "Error: type of data not supported"

################################################################################

def combine_data(x_1, x_2, y_1, y_2):
    """ This is a utility function that simply appends two datasets
    The output will have this format:
    x_res = x_1.T | x_2.T
    y_res = y_1.T | Y_2. T

    The symbol | represents the concatenation and the operation .T represents the transpose
    """
    return np.vstack((x_1,x_2)), np.vstack((y_1.reshape(-1,1),y_2.reshape(-1,1))).ravel()

################################################################################

def display(results):
    """ This function displays the GridSearchCV results in readable format. See search_best()"""
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean,std,params in zip(mean_score,std_score,params):
        print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')

################################################################################
def search_best(x,y, **kwargs):
    """This function uses the GridSearch algorothm to display the best hyperparametrs
    for a given input x and output y. The **kwargs argument specifies the input parameter vectors for
    GridSearch (see @parameters variable inside the function)."""
    model = RandomForestClassifier()
    parameters = {
        "n_estimators": [5, 10, 50, 100, 250],
        "max_depth": [2, 4, 8, 16, 32, None]

    }

    cv = GridSearchCV(model, parameters if not kwargs else kwargs, cv=5)
    cv.fit(x, y)
    display(cv)

def train_summary(model, train_x, train_y, test_x, test_y,  log_file = None):
    """Prints the training and test summary for the @model to the console or to the log file if specified"""
    t_s = time.time()
    model.fit(train_x, train_y)
    t_f = time.time()
    t_e = t_f- t_s

    if log_file:
        logging.basicConfig(filename= log_file, level = logging.INFO)
        logging.info(f"Training finished (time elapsed: {t_e} s")
        logging.info(f"Total Train Instances: {len(train_y)}")
        logging.info(f"Total Test Instances: {len(test_y)}")
        logging.info("Train Results:")
        logging.info("------------------------------------------------------------------------------------------------------------------------")
        logging.info(f"Predicted Anomalies on Train Data: {sum(model.predict(train_x))}")
        logging.info(f"Expected Anomalies on Train Data: {sum(train_y)}")
        logging.info(f"Predicted Anomalies on Test Data: {sum(model.predict(test_x))}")
        logging.info(f"Expected Anomalies on Test Data: {sum(test_y)}")
        logging.info(f"False classifications (sum of False Negative and False Positives) for Train Data: {len(train_y) - sum(model.predict(train_x)== train_y)}")
        logging.info(f"False classifications (sum of False Negative and False Positives) for Test Data: {len(test_y) - sum(model.predict(test_x)== test_y)}")
        logging.info("------------------------------------------------------------------------------------------------------------------------")
        logging.info("Clssification accuracy and errors: (accuracy is [correct predictions] /[total instances], error is [1 - accuracy] ")
        logging.info(f"Train True classification accuracy:{( sum(model.predict(train_x)== train_y))/len(train_y)}%")
        logging.info(f"Train Error: {1- ( sum(model.predict(train_x)== train_y))/len(train_y)}%")
        logging.info(f"Test True classification accuracy:{( sum(model.predict(test_x)== test_y))/len(train_y)}%")
        logging.info(f"Test Error: {1- ( sum(model.predict(test_x)== test_y))/len(train_y)} %")
    else:
        print("------------------------------------------------------------------------------------------------------------------------")
        print(f"Training finished (time elapsed: {t_e} s")
        print(f"Total Train Instances: {len(train_y)}")
        print(f"Total Test Instances: {len(test_y)}")
        print("Train Results:")
        print("------------------------------------------------------------------------------------------------------------------------")
        print(f"Predicted Anomalies on Train Data: {sum(model.predict(train_x))}")
        print(f"Expected Anomalies on Train Data: {sum(train_y)}")
        print(f"Predicted Anomalies on Test Data: {sum(model.predict(test_x))}")
        print(f"Expected Anomalies on Test Data: {sum(test_y)}")
        print(f"False classifications (sum of False Negative and False Positives) for Train Data: {len(train_y) - sum(model.predict(train_x)== train_y)}")
        print(f"False classifications (sum of False Negative and False Positives) for Test Data: {len(test_y) - sum(model.predict(test_x)== test_y)}")
        print("------------------------------------------------------------------------------------------------------------------------")
        print("Clssification accuracy and errors: (accuracy is [correct predictions] /[total instances], error is [1 - accuracy] ")
        print(f"Train True classification accuracy:{( sum(model.predict(train_x)== train_y))/len(train_y)}%")
        print(f"Train Error: {1- ( sum(model.predict(train_x)== train_y))/len(train_y)}%")
        print(f"Test True classification accuracy:{( sum(model.predict(test_x)== test_y))/len(test_y)}%")
        print(f"Test Error: {1- ( sum(model.predict(test_x)== test_y))/len(test_y)} %")






################################################################################
#MAIN
################################################################################
if __name__ == "__main__":

    args = get_args()
    PATH = args.data_path
    files= glob.glob(os.path.join(PATH,"**/*.csv"), recursive = True)
    N = len(files)

    for i,file in enumerate(files):
        file = file.replace("\\","/")
        exec(f"data_{i+1} = pd.read_csv('{file}') ")
        exec(f"x_{i+1},y_{i+1} = process_data(data_{i+1})")

    if args.join:
        for i in range(1,N):
            exec(f"x_1, y_1 = combine_data(x_1, y_1,x_{i+1},y_{i+1})")

    if args.search_best:
        if not args.join:
            for i in range(N):
                exec(f"search_best(x_{i+1}, y_{i+1},\
                n_estimators= [5, 10, 50, 100, 250],\
                min_samples_split= [2, 4, 8, 16, 32, None], \
                    criterion =[ 'entropy', args.criterion],\
                        max_depth = [2, 4, 8, 16, 32, 64])")
        else:
            search_best(x_1, y_1,
                n_estimators= [5, 10, 50, 100, 250],
                min_samples_split= [2, 4, 8, 16, 32, None], \
                    criterion =[ 'entropy', args.criterion],\
                        max_depth = [2, 4, 8, 16, 32, 64])

        sys.exit(0)

    model = RandomForestClassifier( n_estimators= 100, min_samples_split=8, max_depth= 50, random_state= 0)
    train_summary(model, x_1, y_1, x_2, y_2)





