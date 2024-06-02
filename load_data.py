
import pandas as pd
import numpy as np
import random

class DataProcessor:
    def __init__(self, file_name):
        """
        Initializes the DataProcessor with the file name and loads the data.

        Parameters:
        file_name (str): The name of the file to load.
        """
        self.file_name = file_name
        self.df = self.load_data()

    def load_data(self):
        """
        Loads data from a file into a DataFrame. Supports Excel and CSV files.

        Returns:
        DataFrame: The loaded data.
        """
        if ".xlsx" in self.file_name:
            df = pd.read_excel(self.file_name)
            return df
        else:
            df = pd.read_csv(self.file_name)
            return df

    def clean_data(self):
        """
        Cleans the DataFrame by removing columns with names that start with 'Unnamed'.

        Returns:
        DataFrame: The cleaned DataFrame.
        """
        self.df = self.df.filter(regex='^(?!Unnamed)')

    def handle_missing_data(self, subset_column=""):
        """
        Handles missing data by dropping rows with missing values.

        Parameters:
        subset_column (str): The column to consider for missing values. If empty, considers all columns.

        Returns:
        DataFrame: The DataFrame with missing values handled.
        """
        if subset_column == "":
            self.df.dropna(inplace=True)
        else:
            self.df.dropna(subset=[subset_column], inplace=True)

    def handle_duplicates(self, subset_column=""):
        """
        Handles duplicate values in the DataFrame.
    
        Parameters:
        subset_column (str): The column to check for duplicates. If empty, considers all columns.
    
        Returns:
        DataFrame: The DataFrame with duplicates handled.
        """
        if subset_column == "":
            self.df = self.df[~self.df.duplicated()]
        else:
            # Identify duplicates while keeping the first occurrence
            duplicates = self.df[self.df.duplicated(subset=[subset_column], keep='first')]
            
            # Generate new 9-digit numbers for all duplicates
            new_values = {val: random.randint(100000000, 999999999) for val in duplicates[subset_column].unique()}
            
            # Replace duplicates with new 9-digit numbers, excluding the first occurrence
            for index, row in duplicates.iterrows():
                self.df.at[index, subset_column] = new_values[row[subset_column]]
            

    def add_labeled_data(self, label_data, merge_column):
        """
        Adds labels to the DataFrame by merging with another DataFrame.

        Parameters:
        label_data (DataFrame): The DataFrame containing the labels.
        merge_column (str): The column name to use for merging.

        Returns:
        DataFrame: The merged DataFrame with labels added.
        """
        label_df = pd.DataFrame(label_data)
        self.df = pd.merge(self.df, label_df, how='left', left_on=merge_column, right_on=merge_column)
        

    def add_multiple_labels(self, label_data_dict):
        """
        Adds multiple labels to the DataFrame by merging with several label DataFrames.

        Parameters:
        label_data_dict (dict): A dictionary where keys are column names and values are label DataFrames.

        Returns:
        DataFrame: The DataFrame with all labels added.
        """
        for merge_column, label_data in label_data_dict.items():
            self.add_labeled_data(label_data, merge_column)
        
