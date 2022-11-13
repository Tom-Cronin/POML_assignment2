import pandas as pd

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
