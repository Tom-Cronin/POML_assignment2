import pandas as pd

def read_data_return_dataframe(PathToFile):
    """
    Reads data from a specified text file. Requires the path to the text file

    Should return a dataframe of the test file
    >>> type(read_data_return_dataframe('testdata.txt'))
    <class 'pandas.core.frame.DataFrame'>

     Should have correct data in the dataframe.
     There should be 20 entrys in the dataframe excluding of the column names
    >>> read_data_return_dataframe('testdata.txt').size
    20
    """
    return pd.read_table(PathToFile) # reads txt file and converts it to a pandas dataframe


def convert_label(dataframe, label, old_values, new_values):
    ndarray = dataframe[label].copy()
    for index in range(len(ndarray)):
        if old_values[0] in ndarray[index].lower():
            ndarray[index] = new_values[0]
        elif old_values[1] in ndarray[index].lower():
            ndarray[index] = new_values[1]
    dataframe[label] = ndarray
    return dataframe


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