import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

def download_chicago_crime_dataset(start_year=2001):
    """
    Download Chicago Crime Dataset from the Chicago Data Portal.
    The dataset is available from 2001 to present.

    Parameters
    start_year : int
        The year to start downloading the dataset. Default is 2001.
    """

    # Create Directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    URL = 'https://data.cityofchicago.org/resource/ijzp-q8t2.csv'
    current_year = datetime.now().year 

    # Fetching Data for each year
    for year in range(start_year, current_year+1):

        query = f"?$where=year={year}"
        response = requests.get(URL+query)

        print(f'Downloading data for year {year} | Status Code: {response.status_code}')
        
        if(response.status_code == 200):
            filename = f'data/raw/chicago_crimes_{year}.csv'
            with open(filename,'wb') as f:
                f.write(response.content)
                print('Successfully created dataset for {year}')
        else:
            print(f'Dataset Creation Failed for {year}')

def download_census_data():
    """
    TODO
    """
    pass

def main():
        download_chicago_crime_dataset()
        
if __name__ == '__main__':
    load_dotenv()
    main()
        


