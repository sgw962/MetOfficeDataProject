from dotenv import load_dotenv, find_dotenv
import os
from src.data.data_upload import DataPoint
from src.models.weather_classifier import WeatherClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


def main():
    #Retrieve API
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    api_key = os.environ.get('API')
    #Instantiate DataPoint with API
    data = DataPoint(api_key)

    #Call get_location_id with 'Southampton' and print resulting ID
    location = data.get_location_id('Southampton')
    print(f'Location ID for Southampton: {location}')

    #Sample 30 location IDs
    location_ids = ['310012', '310009', '310004', '310042', '310046', '310013', '351905', '310069', '310016',
                    '14', '26', '33', '3006', '3068', '3075', '3081', '3002', '3005', '310024', '310022',
                    '310025', '310047', '3908', '310035', '310026', '310048', '310031', '310037', '310011',
                    '310006']

    #Call get_forecast_data with IDs. Print result then save to csv
    forecast_data = data.get_forecast_data(location_ids)
    print(forecast_data)
    data.save_to_csv(forecast_data, file_name='Raw_Data.csv', folder_path='/Users/seanwhite/OneDrive - University of Greenwich/Documents/Advanced Programming/Coursework/CourseworkReport&Code/data/raw/')
    #Call fix_columns with result of get_forecast_data. Print result then save to csv
    processed_data = data.fix_columns(forecast_data)
    print(processed_data)
    data.save_to_csv(processed_data, file_name='Processed_Data.csv', folder_path='/Users/seanwhite/OneDrive - University of Greenwich/Documents/Advanced Programming/Coursework/CourseworkReport&Code/data/processed/')

    #Instantiate WeatherClassifier with API and location_ids list
    weather = WeatherClassifier(api_key, location_ids)
    #Produce classification report for RandomForestClassifier
    weather.train_and_print_report(RandomForestClassifier, 'Random Forest Classifier')
    #Produce classification report for GradientBoostingClassifier
    weather.train_and_print_report(GradientBoostingClassifier, 'Gradient Boosting Classifier')


#File won't run if being imported by another
if __name__ == '__main__':
    main()
else:
    print('main.py file is not being used correctly - should only be run directly not imported elsewhere')
