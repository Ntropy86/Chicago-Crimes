from multiprocessing import process
import sys
import pandas as pd
import numpy as np
from numpy import dtype
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
    start_ywar (int): Starting year for processing 
    expected_columns (dict): Key value pairs of the column names for chicago crimes dataset and it's expected datatype.
    """

    def __init__(self,raw_data_path:str='data/raw', processed_data_path:str='data/processed', start_year:int=2001) -> None:
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path

        # Define expected columns and their types
        self.expected_columns = {
            'id': 'str',
            'case_number': 'str',
            'date': 'datetime64[ns]',
            'block': 'str',
            'primary_type': 'category',
            'description': 'category',
            'location_description': 'category',
            'arrest': 'bool',
            'domestic': 'bool',
            'beat': 'str',
            'district': 'str',
            'ward': 'str',
            'community_area': 'str',
            'latitude': 'float64',
            'longitude': 'float64'
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

            df = pd.read_csv(file_path, low_memory=False)

            logger.debug(f'Sucessfully loaded {len(df)} records for {year}')
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
            # logger.debug(f'Df Description: {df.describe()}')
            # logger.debug(f'Df Info: {df.info()}')

            # df['Date'] = pd.to_datetime(df['Date'])
            # df['Primary Type'] = df['Primary Type'].astype('category')
            # df['Location Description'] = df['Location Description'].astype('category')
            # df['Arrest'] = df['Arrest'].astype(bool)
            # df['Domestic'] = df['Domestic'].astype(bool)

            missing_cols = set(self.expected_columns.keys()) - set(df.columns)
            
            if missing_cols:
                raise DataProcessingError(f'Missing a couple of Expected Columns. Are you sure you have the right Dataset? Missing {missing_cols}')

            # For automated Conversion of Data Types Pandas requires this
            dtype_mapping = {
            'str': str,
            'category': 'category',  # pandas understands 'category' directly
            'bool': bool,
            'float64': float,
            'int64': int,
            'datetime64[ns]': 'datetime64[ns]'
        }
                                    
            for col, dtype_name in self.expected_columns.items():
                if dtype_name == 'datetime64[ns]':
                    df[col] = pd.to_datetime(df[col])
                else:
                    actual_type = dtype_mapping.get(dtype_name)
                    if actual_type:
                        df[col] = df[col].astype(actual_type)
                    else:
                        logger.warning(f"Unknown dtype {dtype_name} for column {col}")


            
            # Pandas 3.0 Future Warning Removal : DO NOT USE inplace as it is DEPRECATED 
            df['latitude'] = df['latitude'].fillna(df['latitude'].mean())
            df['longitude'] = df['longitude'].fillna(df['longitude'].mean())

            df = df.drop_duplicates(subset=['id'], keep='first')

            df = df.drop(['x_coordinate','y_coordinate'], axis=1)

            logger.debug(f'Cleaned {len(df)} records!')

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
        
    def _engineer_features(self,df:pd.DataFrame) -> pd.DataFrame:
        """
        Creates new features from the existing dataset

        Args:
        df (pd.DataFrame): Dataframe containing cleaned data 

        Returns:
        pd.DataFrame: Dataframe with new features added

        Raises:
        DataProcessingError : Error Raised in Feature Engineering. See message for more details
        """

        logger.info('Starting Feature Engineering Process!')
        try:
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['hour'] = df['date'].dt.hour
            df['day_of_week'] = df['date'].dt.dayofweek

            bins = [0,6,12,15,18,22,24]
            labels = ['Night','Morning','Afternoon','Evening','LateEvening','Night'] # 2 Nights so that we wrap around the Night time (22 -> 06)
            df['time_of_day'] = pd.cut(df['hour'],bins=bins,labels=labels, ordered=False) # Ordered False as we have a Wrap around 
            # Time of day produces some null data points keep that in mind

            df['has_location'] = df['location'].notna()
            df = df.drop(['location'],axis=1)

            df['block'] = df['block'].str[5:] # Removed any unecessary sensitive information that is of no use to us.

            logger.debug(f'Feature Engineering Sucessfully Completed!')
            return df
        except Exception as e:
            logger.error(f'Failed to Create new features. {str(e)}')
            raise DataProcessingError(f'Failed to Create new features. {str(e)}')
        
    def _process_year(self, year:int) -> None:
        """
        Processes the Raw Dataset for the year provided.
        - Cleans the Data
        - Engineers Features
        - Saves the processed data in self.processed_data_path folder

        Args:
        year (int): Year of the Data Report to be Processed

        Returns:
        None

        Raises:
        DataProcessingError : If any of the above methods fail, this raises an Exception.
        """
        try:
            df = self._load_raw_data(year)
            df = self._clean_data(df)
            df = self._engineer_features(df)

            output_path = self.processed_data_path + '/' + f'chicago_crimes_{year}_processed.csv'
            df.to_csv(output_path,index=False)
            logger.info(f'Sucessfully Processed {len(df)} records and saved as {output_path}')
            del(df)

        except Exception as e:
            logger.error(f'MAJOR: Cannot Process Data. {str(e)}')
            raise DataProcessingError

        
def main():
    """
    Main Execution function for Data Cleaning

    Important:
    Make sure to run this code as python -m src.data.preprocessing <start_year> otherwise defaults to 2001
    """
    try:
        if len(sys.argv)>1:
            start_year = int(sys.argv[1])
        else:
            start_year = 2001

        logger.info(f'Starting Preprocessing of All Chicago Crimes Dataset. Start year is {start_year}')
        preprocessor = CrimeDataProcessor(raw_data_path='data/raw',processed_data_path='data/processed',start_year=start_year)
        current_year = datetime.now().year

        for year in range(start_year,current_year+1):
            preprocessor._process_year(year)
        

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



    

