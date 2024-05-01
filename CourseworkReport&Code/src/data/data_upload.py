import pandas as pd
import requests
import os
from dotenv import load_dotenv, find_dotenv
from src.utilities import weather_mapping


class DataPoint:
    """
    Initiate this class with a valid met office API. It will upload the data using json requests
    and has the functionality of:
    - Returning a location ID for a given name
    - Returning a dataframe of provided weather information for given location IDs
    - Saving a dataframe to csv format
    - Fixing the format and presentation of the dataframe
    """

    def __init__(self, API):
        """
        Initialises the class establishing the API, fixed elements of the URL and location IDs to be used in
        subsequent methods.
        """
        self.API = API
        # Load URL elements from .env file
        dotenv_path = find_dotenv()
        load_dotenv(dotenv_path)
        self.Base = os.environ.get("BaseURL")
        self.Endpoint = os.environ.get("EndpointURL")

    def get_location_id(self, location_name):
        """
        Call this function with a valid location name from the met office data for which it will return the
        corresponding location id.
        """
        try:
            response = requests.get(f'{self.Base}sitelist{self.Endpoint}{self.API}')
            if response.status_code == 200:
                data = response.json()
                for location in data['Locations']['Location']:
                    if location['name'] == location_name:
                        return location['id']
                raise ValueError(f'Location "{location_name}" not found - check this is a valid name')
            else:
                print(f'Error: status code is {response.status_code}')
        except Exception as e:
            print(f'Error: {e}')

    def get_forecast_data(self, location_ids):
        """
        This function will, using a list of location ids, return a dataframe with the 3hourly, 5-day forecast provided
        by the met office for each of the locations. There is no need for any user input to call this function.
        """
        forecast_data = pd.DataFrame()
        for location_id in location_ids:
            try:
                response = requests.get(f'{self.Base}{location_id}{self.Endpoint}{self.API}')
                if response.status_code == 200:
                    data = response.json()
                    location_name = data['SiteRep']['DV']['Location']['name']
                    for day_data in data['SiteRep']['DV']['Location']['Period']:
                        forecast = day_data['Rep']
                        forecast_df = pd.DataFrame(forecast)
                        forecast_df.insert(0, 'Location', location_name)
                        forecast_df.insert(1, 'Day', day_data['value'])
                        forecast_data = pd.concat([forecast_data, forecast_df], ignore_index=True)
                else:
                    print(f'error: status code is {response.status_code}')
                    print('Location ID(s) may not be valid')
            except Exception as e:
                print(f'Error fetching data for location {location_id}: {e}')
                print('Try checking URL elements are correct')
        return forecast_data

    def save_to_csv(self, data, file_name, folder_path):
        """
        Saves data as a csv file in the given folder. Please call this function with the following:
        - A pandas dataframe
        - A name for the csv file
        - A path to the folder you'd like to store the csv file in
        """
        try:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            file_path = os.path.join(folder_path, file_name)
            data.to_csv(file_path, index=False)
            print(f'{file_name} successfully saved to {folder_path}')
        except Exception as e:
            print(f'Error saving data to CSV: {e}')
            print('Check folder path exists & file name ends in ".csv"')

    def fix_columns(self, df):
        """
        Call this function with the dataframe with which it:
        - Correct the column names to a readable format
        - Removes the unnecessary "$" column
        - Simplify the "Weather Type" column by concatenating similar values in to one
        """
        try:
            df.rename(columns={'D': 'Wind Direction', 'F': 'Feels Like Temperature', 'G': 'Wind Gust',
                               'H': 'Screen Relative Humidity', 'Pp': 'Precipitation Probability', 'S': 'Wind Speed',
                               'T': 'Temperature', 'V': 'Visibility', 'W': 'Weather Type', 'U': 'Max UV Index'},
                      inplace=True)
        except Exception as e:
            print(f'Error renaming columns: {e}')
            print('Data used to initiate function may be incompatible with this function')
        if '$' in df.columns:
            del df['$']
        else:
            pass
        try:
            df['Weather Type'] = df['Weather Type'].map(weather_mapping)
            return df
        except Exception as e:
            print(f'Error mapping weather types: {e}')
            print('Data being used may be incompatible with "weather_mapping" template')
