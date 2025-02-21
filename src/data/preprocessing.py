from multiprocessing import process
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


# === Configs ===
try:
    from src.utils.logger_config import setup_logger
    from src.utils.exceptions import DataProcessingError

    logger = setup_logger(__name__)
except Exception as e:
    print(f'CRITICAL ERROR: download_data : Logger Config issue : {str(e)}')
    exit(1)

# === END Configs ===

class CrimeDataProcessor:
    """
    A class to handle the preprocessing of the Chicago Crime Dataset Downloaded from src.data.download_data

    This class handles:
    - Loading raw CSV files
    - Cleaning and Standardizing data
    - Feature Engineering
    - Saving the Preprocessed Data

    Attributes:
    raw_data_path (str): Path for the raw data folder. Defaults to 'data/raw'
    processed_data_path (str): Path of the folder for processed data to be stored. Defaults to 'data/processed' 
    expected_columns (dict): Key value pairs of the column names for chicago crimes dataset and it's expected datatype.
    """

    def __init__(self,raw_data_path:str='data/raw', processed_data_path:str='data/processed') -> None:
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path

        # Define expected columns and their types
        self.expected_columns = {
            'ID': 'str',
            'Case Number': 'str',
            'Date': 'datetime64[ns]',
            'Block': 'str',
            'Primary Type': 'category',
            'Description': 'category',
            'Location Description': 'category',
            'Arrest': 'bool',
            'Domestic': 'bool',
            'Beat': 'str',
            'District': 'str',
            'Ward': 'str',
            'Community Area': 'str',
            'Latitude': 'float64',
            'Longitude': 'float64'
        }
    
    def _load_raw_data(self,year) -> pd.DataFrame:
        """
        Loads the Raw Data from a CSV to a Pandas DataFrame

        Args:
        year (int): Year of the dataset to load

        Returns:
        pd.DataFrame: The dataframe containing all the values from the csv

        Raises:
        DataProcessingError: Error with handling the csv. The message shall give more insights
        """ 
        try:
            file_path = self.raw_data_path + '/' + f'chicago_crimes_{year}.csv'
            logger.info(f'Processing {file_path}')

            df = pd.read_csv(file_path)
            logger.debug(f'Sucessfully loaded {len(df)} for {year}')
            return df
        except Exception as e:
            logger.info(f'Error in Loading for {year} : {str(e)}')
            raise DataProcessingError(f'Error in Loading for {year} : {str(e)}')
    
    def _clean_data(self,df:pd.DataFrame) -> pd.DataFrame:
        """
        Data Cleaning Function 
        - Cleans the Raw Dataset
        - Standardizes the Values when needed

        Args:
        df (pd.DataFrame): Dataframe containing the Loaded Data

        Returns:
        pd.DataFrame: Cleaned Dataframe

        Raises:
        DataProcessingError : Error Raised in Cleaning Data. See message for more details
        """

        try:
            logger.info('Starting Data Cleaning Process')

            # Analysis 
            logger.debug(f'Df Description: {df.describe()}')
            logger.debug(f'Df Info: {df.info()}')

            # change [date, updated on]  to datetime 
            # Including Non Nulls : 
            # NEW COL: slice Block column to remove building number
            # Visualization : [(lat,long) ]
            # Groupby potential: [iucr, new_block_column , year ]
            # Removal : [x,y coords, updatedon?, location?]
            # Category cols?



            return df
        except Exception as e:
            logger.error(f'Error in Cleaning Data : {str(e)}')
            raise DataProcessingError(f'Error in Cleaning Data : {str(e)}')
        
def main():
    """
    Main Execution function for Data Cleaning
    """
    try:
        data_cleaner = CrimeDataProcessor(raw_data_path='data/raw',processed_data_path='data/processed')
        df = data_cleaner._load_raw_data('2021') # Aribtary currently to understand the dataset in question.
        df = data_cleaner._clean_data(df)
    except Exception as e:
        logger.error(f'Error in Running Application. {str(e)}')
        raise DataProcessingError(f'Error in Running Application. {str(e)}')

if __name__ == '__main__':
    try:
        logger.info('Launching Data Cleaner!')
        main()
    except Exception as e:
        logger.critical(f'Application Failure. {str(e)}')
        exit(1)



    

