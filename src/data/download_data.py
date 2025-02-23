"""
Chicago Crime Dataset Downloader Module.

This module handles the download of crime data from Chicago's Open Data Portal using the SODA API.
It implements parallel processing for improved download speeds and follows proper rate limiting
to respect API constraints.

Note:
    SODA (Socrata Open Data API) is the API framework used by many government open data portals,
    including Chicago's data portal. More information can be found at:
    https://dev.socrata.com/
"""


import sys
from urllib import response
from numpy import full
import requests
import os
from io import StringIO
from typing import Optional, Tuple, List
from datetime import date, datetime
from dotenv import load_dotenv
import time
import pandas as pd
import multiprocessing as mp
from pathlib import Path



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
    A class to handle parallel download of Raw Chicago Crimes Dataset.

    This class handles downloading crime data from Chicago's Open Data Portal using
    parallel processing for improved performance. It implements proper rate limiting
    and error handling.

    Attributes:
        BASE_URL (str): Base URL for the SODA API endpoint
        start_year (int): Year to start downloading the data from
        rate_limit (float): Time to wait between requests in seconds
        batch_size (int): Number of records to fetch per request
        max_workers (int): Maximum number of parallel processes to use
        
    Example:
        >>> downloader = ChicagoCrimeDatasetDownloader(start_year=2001)
        >>> downloader.download_dataset()
    """

    def __init__(
        self,
        max_workers:int,
        start_year: int=2001, 
        rate_limit:float=1.0 ,
        batch_size:int=1000,
        ) -> None:
        """
        Initalize the Downloader with configs as mentioned in the class description

        Args:
            start_year (int): Year to start downloading the crime Dataset from
            rate_limit (float): Time to wait between creating the requests in seconds
        """
        self.URL = 'https://data.cityofchicago.org/resource/ijzp-q8t2.csv'
        self.start_year = start_year
        self.rate_limit = rate_limit
        self.batch_size = batch_size
        self.max_workers = max_workers

        # Create the Directories Required 
        self._create_directories()

    def _create_directories(self) -> None:
        """
        Creates the necessary Folders to store the data
        """
        try:
            os.makedirs('data/raw',exist_ok=True)
            os.makedirs('data/processed',exist_ok=True)
            os.makedirs('data/temp', exist_ok=True)
            logger.debug("Data Directories successfully created!")
        except Exception as e:
            logger.error(f'Failed to Create Directories. {str(e)}')
            raise ConfigError(f'Failed to Create Directories. {str(e)}')

    @staticmethod
    def _construct_date_query(year:int,month:int) ->str:
        """
        Constructs a SODA API style query for the specific year and month for download

        Args:
            year (int): Target year for data download
            month (int): Target month (1-12)
        
        Returns:
            str: Formatted Query String for the SODA API
        """
        start_date = datetime(year,month,1)
        if(month==12):
            end_date = datetime(year+1,1,1)
        else:
            end_date = datetime(year,month+1,1)
        
        return f'?$where=date>="{start_date.strftime("%Y-%m-%d")}" AND date < "{end_date.strftime("%Y-%m-%d")}"'

    def _make_request(
            self,
            query:str,
            offset:int=0,
            retries:int=3

            ) -> Optional[requests.Response]:
        """
        Make an HTTP request to the SODA API with retry logic.

        Args:
            query (str): The query string for the API
            offset (int): Number of records to skip for pagination (Default = 0)
            retries (int): Number of times to retry failed requests (Default = 3)

        Returns:
            Optional[requests.Response]: Response object if successful, None otherwise
        """

        full_url = f"{self.URL}{query}&$limit={self.batch_size}&$offset={offset}"
    
        for attempt in range(retries):
            try:
                logger.debug(f'Requesting: {full_url}')
                response = requests.get(full_url, timeout=30)

                if response.status_code == 200:
                    return response
                
                elif response.status_code == 429:  # Rate limit
                    wait_time = min((attempt + 1) * self.rate_limit * 2, 60)
                    logger.warning(f'Rate Limit Exceeded, waiting for {wait_time}s')
                    time.sleep(wait_time)
                else:
                    logger.error(f'Request failed with status {response.status_code}: {response.text}')
                    time.sleep(self.rate_limit * (attempt + 1))  # Progressive backoff

            except requests.exceptions.RequestException as e:
                logger.error(f'Network error (attempt {attempt + 1}): {str(e)}')
                if attempt == retries - 1:
                    raise DatasetDownloadError(f'Network error after {retries} attempts: {str(e)}')
                time.sleep(self.rate_limit * (attempt + 1))
        
        return None
    
    def _download_month_data(self, year_month: Tuple[int, int]) -> None:
        """
        Download data for a specific year and month.

        Args:
            year_month: Tuple of (year, month) to download
        """
        year, month = year_month
        monthly_chunks = []
        offset = 0
        total_records = 0
        
        try:
            # Keep downloading chunks until we get an empty response
            while True:
                query = self._construct_date_query(year, month)
                response = self._make_request(query, offset)
                
                if not response:
                    raise Exception(f'Failed to get response from the Server for {month}/{year}')
                
                chunk_df = pd.read_csv(StringIO(response.text))
                
                if chunk_df.empty:
                    # If this is our first chunk and it's empty, there's no data for this month
                    if offset == 0:
                        logger.info(f"No data for {year}-{month:02d}")
                        return
                    # If we've gotten data before and now get an empty chunk, we're done
                    break
                    
                monthly_chunks.append(chunk_df)
                total_records += len(chunk_df)
                offset += self.batch_size
                time.sleep(self.rate_limit)

            # Save the data - OUTSIDE the while loop
            if monthly_chunks:  # Only try to save if we got any data
                try:
                    monthly_df = pd.concat(monthly_chunks, ignore_index=True)
                    temp_path = f'data/temp/chicago_crimes_{year}_{month:02d}.csv'
                    monthly_df.to_csv(temp_path, index=False)
                    logger.info(f"Saved {total_records} records for {year}-{month:02d}")
                except Exception as e:
                    logger.error(f"Failed to save data for {year}-{month:02d}: {str(e)}")
                    raise

        except Exception as e:
            logger.error(f'Failed to fetch data for {year}-{month:02d}: {str(e)}')
            raise DatasetDownloadError(f'Failed to fetch data for {year}-{month:02d}: {str(e)}')
    
    def _process_year(self,year:int) -> None:
        """
            Process and combine monthly data files for a specific year.

            Args:
                year: The year to process
            
            Raises:
                DataDownloadError : See the logs for further Diagnosis of the error in download.
            """
        try:
            temp_dir = Path('data/temp')
            monthly_files = list(temp_dir.glob(f'chicago_crimes_{year}_*.csv'))
            
            if not monthly_files:
                raise Exception('Couldn\'t find monthly files')
                
            # Combine all monthly files
            yearly_chunks = []
            for file_path in monthly_files:
                df = pd.read_csv(file_path)
                yearly_chunks.append(df)
                file_path.unlink() # Delete File
                
            if yearly_chunks:
                yearly_df = pd.concat(yearly_chunks, ignore_index=True)
                output_path = f'data/raw/chicago_crimes_{year}.csv'
                yearly_df.to_csv(output_path, index=False)
                logger.info(f"Combined {len(yearly_df)} records for {year}")

            if temp_dir.exists():
                # Remove all files in temp dir
                for file in temp_dir.iterdir():
                    file.unlink()
                # Remove directory itself
                temp_dir.rmdir()
                logger.debug(f"Cleaned up temporary directory")
                
        except Exception as e:
            logger.error(f"Failed to process year {year}: {str(e)}")
            raise DatasetDownloadError(f"Failed to process year {year}: {str(e)}")       
    


    def _download_dataset(self) -> None:
        """
        Downloads the Chicago Crime Dataset for all the years from start_year till present
        """
        try:
            current_year = datetime.now().year
            
            # Create download tasks
            download_tasks = [
                (year, month)
                for year in range(self.start_year, current_year + 1)
                for month in range(1, 13)
            ]

            # Process in batches to control concurrent requests
            batch_size = min(self.max_workers * 2, len(download_tasks))  # Control batch size
            for i in range(0, len(download_tasks), batch_size):
                batch = download_tasks[i:i + batch_size]
                
                with mp.Pool(self.max_workers) as pool:
                    # Use map for synchronous execution - more controlled than apply_async
                    pool.map(self._download_month_data, batch)
                    pool.close()
                    pool.join()
                
                logger.info(f"Completed batch {i//batch_size + 1} of {(len(download_tasks) + batch_size - 1)//batch_size}")
                time.sleep(self.rate_limit)  # Rate limit between batches

            # Process years only after all months are downloaded
            successful_years = []
            for year in range(self.start_year, current_year + 1):
                try:
                    self._process_year(year)
                    successful_years.append(year)
                    logger.info(f"Successfully processed year {year}")
                except Exception as e:
                    logger.error(f'Failed to process year {year}: {str(e)}')
                    continue

            if successful_years:
                logger.info(f'Successfully processed years: {successful_years}')
            else:
                logger.warning('No years were successfully processed')
                
        except Exception as e:
            logger.error(f'Error in dataset download: {str(e)}')
            raise DatasetDownloadError(f'Error in dataset download: {str(e)}')
                

def download_census_data() -> int:
    """
    Downloads the Census data for Chicago
    TODO: Implement the census data download functionality
    """
    return 0

def main():
    """
    Main execution function for Dataset Download
    """
    try:
        if len(sys.argv)>1:
            start_year = int(sys.argv[1])
        else:
            start_year = 2001
        logger.info('Running Dataset Downloader for Chicago Crimes')
        # Calculate optimal workers based on CPU count and system resources
        max_workers = min(mp.cpu_count() - 1 or 1, 1000)  
        
        downloader = ChicagoCrimeDatasetDownloader(
            max_workers=max_workers,
            start_year=start_year,
            rate_limit=0.2,
            batch_size=1000
        )
        downloader._download_dataset()
        logger.info('All Datasets Downloaded for Chicago Crimes')

        logger.info('Starting Census Dataset Download')
        implement_flag = download_census_data()
        if implement_flag:
            logger.info('Census Data Download Complete')
        else:
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

        


