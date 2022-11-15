import pandas as pd
import numpy as np

# Tom Cronin
def read_data_return_dataframe(PathToFile):
    """
    Reads data from a specified text file. Requires the path to the text file
    """
    return pd.read_table(PathToFile) # reads txt file and converts it to a pandas dataframe

# Daniel Verdejo - split into labels and attributes
def split_df_labels_attributes(df):
    """ Split the dataframe into two by labels and attributes

        Keyword arguments:
        df -- A pandas dataframe type containing labels and attributes
        label_col_name -- A string which contains the name of the label column. 

        Returns:
        tuple -- (label: pd.DataFrame, attributes: pd.DataFrame)
      """
    return (df.iloc[:,0:1], df.iloc[:,1:])  # (for every row take columns upto index 1 exclusive, for every row take every column from 1 onwards inclusive)

# Daniel Verdejo - train, test splilt dataframe
def split_df_to_train_test_dfs(df, test_set_size=.1, random_state=42):
    """ Splits a single dataframe into 2 dataframes
    
    Arguments:
    df -- A pandas Dataframe to be split into 2
    test_set_size -- upper limit of the split. Defaults to 10%

    Returns:
    tuple -- (X_train, X_test, y_train, y_test)
    """
    df_copy = df.sample(frac = 1) # shuffle data frame

    df_train = df_copy.sample(frac = 1 - test_set_size, random_state=random_state) # randomly sample a fraction of the dataframe between 60 & 70 % of its entirety
    df_test = df_copy.drop(df_train.index) 
    
    y_train, X_train = split_df_labels_attributes(df_train)
    y_test, X_test = split_df_labels_attributes(df_test)
    return (X_train, X_test, y_train, y_test) # return the training data and the test data


# Tom Cronin ToDo coment and add tests
# TODO: Add tests
def Normalize(ndarray, features):
    """
    Standardises the data by subtracting the mean from each element and dividing it by the standard deviation
    The results of standarising the data will reduce the standard deviation to 1 and the mean of each feature to 0
    :param ndarray: A multidimentional array containing the features needed to be normalised
    :param features: A list of features in the ndarray e.g. Wind, Year
    :return: returns a new ndarry containg standardised data

    ToDo: Add tests
    """
    for feature in range(len(features)):
        array = ndarray[:, feature] # gets column of feature
        xmin = min(array)      # gets the min value
        xmax = max(array)      # gets the max value
        ndarray[:, feature] = (array-xmin)/(xmax-xmin)  # calculates and replaces the column with its normalised version
    return ndarray

def convert_label(dataframe, label, old_values, new_values):
    ndarray = dataframe[label].copy()
    for index in range(len(ndarray)):
        if old_values[0] in ndarray[index].lower():
            ndarray[index] = new_values[0]
        elif old_values[1] in ndarray[index].lower():
            ndarray[index] = new_values[1]
    dataframe[label] = ndarray
    return dataframe
