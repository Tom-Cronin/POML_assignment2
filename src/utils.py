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
def split_df_to_train_test_dfs(df, lower=0.66, upper=0.9):
    """ Splits a single dataframe into 2 dataframes
    
    Arguments:
    df -- A pandas Dataframe to be split into 2
    lower -- lower limit of the split. Defaults to 66%
    upper -- upper limit of the split. Defaults to 90%

    Returns:
    tuple -- (df_train: pandas.Dataframe, df_test: pandas.Dataframe)
    """
    train_frac = round(np.random.uniform(lower, upper), 2) # get a random float for our training fraction
    df_train = df.sample(frac = train_frac) # randomly sample a fraction of the dataframe between 60 & 70 % of its entirety
    return (df_train,  df.drop(df_train.index)) # return the training data and the test data


# Tom Cronin ToDo coment and add tests
# TODO: Add tests
def normalise(ndarray, features):
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
