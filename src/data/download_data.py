from json import load
import os
from tkinter import E
from h11 import Response
from numpy import full
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional
import time

# === Configs ===
try:
    from src.utils.logger_config import setup_logger
    from src.utils.exceptions import DatasetDownloadError,ConfigError
    logger = setup_logger(__name__)
    logger.info('Logger successfully Initalized')
except Exception as e:
    print(f'CRITICAL ERROR: download_data : Logger Config issue : {str(e)}')
    exit(1)

# === END Configs ===

class ChicagoCrimeDatasetDownloader:
    """
     A class to handle Download of Raw Chicago Crimes Dataset for each year

     Attributes:
     URL (str) : Base URL for the API Portal for Chicago Crimes
     start_year (int) : Year to start downloading the data
     rate_limit (float) : Time to wait in between requests in seconds
    """

    def __init__(self, start_year: int = 2001, rate_limit:float = 1.0 ) -> None:
        """
        Initalize the Downloader with configs as mentioned in the class description

        Args:
            start_year (int): Year to start downloading the crime Dataset from
            rate_limit (float): Time to wait between creating the requests in seconds
        """
        self.URL = 'https://data.cityofchicago.org/resource/ijzp-q8t2.csv'
        self.start_year = start_year
        self.rate_limit = rate_limit

        # Create the Directories Required 
        self._create_directories()

    def _create_directories(self) -> None:
        """
        Creates the necessary Folders to store the data
        """
        try:
            os.makedirs('data/raw',exist_ok=True)
            os.makedirs('data/processed',exist_ok=True)
            logger.debug("Data Directories successfully created!")
        except Exception as e:
            logger.error(f'Failed to Create Directories. {str(e)}')
            raise ConfigError(f'Failed to Create Directories. {str(e)}')

    def _make_request(self,year:int) -> Optional[requests.Response]:
        """
        Makes an HTTP request to download the data for the year specified

        Args:
            year (int): The year data is to be requested for
        
        Returns:
            requests.Response: Response obj if successful
        
        Raises:
            DataDownloadError: If the request fails
        """

        query = f'?$where=year={year}'
        full_URL = self.URL + query

        try:
            logger.info(f'Downloading Data for {year}')
            response = requests.get(full_URL, timeout=10)
            
            if response.status_code == 200:
                logger.debug(f'Sucessfully Downloaded Data for {year}')
                return response
            else:
                logger.error(f'Error Downloading Data for {year} : Status Code:{response.status_code} Message: {response.content}')
                return None
        except Exception as e:
            logger.error(f'Request failed for year {year} : {str(e)}')
            raise DatasetDownloadError(f'Request failed for year {year} : {str(e)}')

    def _download_dataset(self) -> None:
        """
        Downloads the Chicago Crime Dataset for all the years from start_year till present
        """
        current_year = datetime.now().year
        
        for year in range(self.start_year,current_year+1):
            try:
                response = self._make_request(year)

                if response:
                    filename = f'data/raw/chicago_crimes_{year}.csv'
                    with open(filename,'wb') as f:
                        f.write(response.content)
                    logger.info(f"Successfully created Raw Data for {year}")
                
                time.sleep(self.rate_limit)
            except Exception as e:
                logger.error(f'Error in Creating Raw Dataset for {year}. {str(e)}')
                continue # Continue to the next year processing
                

def download_census_data() -> int:
    """
    Downloads the Census data for Chicago
    TODO: Implement the census data download functionality
    """
    return 0

def main():
    """
    Main execution function
    """
    try:
        logger.info('Running Dataset Downloader for Chicago Crimes')
        downloader = ChicagoCrimeDatasetDownloader(start_year=2001)
        downloader._download_dataset()
        logger.info(f'All Datasets Downloaded for Chicago Crimes')

        logger.info('Starting Census Dataset Download')
        implement_flag =download_census_data()
        if(implement_flag):
            logger.info('Census Data Download Complete')
        else : 
            logger.warning('Census Dataset Download function is yet to be implemented.')
    
    except Exception as e:
        logger.error(f'Dataset Downloader Failed! {str(e)}')
        raise 
        


if __name__ == '__main__':
    try:
        load_dotenv()
        main()
    except Exception as e:
        logger.critical(f'Application Terminated: {str(e)}')
        exit(1)

        


