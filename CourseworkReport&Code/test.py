import pytest
import pandas as pd
from python-dotenv import load_dotenv, find_dotenv
from src.data.data_upload import DataPoint
from src.models.weather_classifier import WeatherClassifier
import os
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from src.utilities import weather_dict

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
api_key = os.environ.get('API')


@pytest.mark.parametrize('location_name, expected_id', [('Baltasound', '3002'),
                                                        ('Southampton', '353595'),
                                                        ('Crosby', '3316')])
def test_get_location_id(location_name, expected_id):
    assert DataPoint(api_key).get_location_id(location_name) == expected_id


@pytest.mark.parametrize('location_name', [('Cork'),
                                           ('Dublin'),
                                           ('New York'),
                                           ])
def test_false_id(location_name, capsys):
    DataPoint(api_key).get_location_id(location_name)
    captured = capsys.readouterr()
    assert f'Error: Location "{location_name}" not found - check this is a valid name\n' in captured.out


@pytest.mark.parametrize('location', [('3002'),
                                      ('353595'),
                                      ('3316')])
def test_get_forecast_data(location):
    forecast_data = DataPoint(api_key).get_forecast_data(location)
    assert isinstance(forecast_data, pd.DataFrame)
    assert len(list(forecast_data.values)) > 0
    assert list(
        forecast_data.columns) == ['Location', 'Day', 'D', 'F', 'G', 'H', 'Pp', 'S', 'T', 'V', 'W', 'U', '$']


def test_false_forecast_data():
    not_forecast_data = DataPoint(api_key).get_forecast_data(['33333333'])
    assert len(list(not_forecast_data.values)) == 0
    assert not_forecast_data.empty


@pytest.mark.parametrize("data, file_name, folder_path", [
    (pd.DataFrame({'col1': [4, 5, 6], 'col2': ['d', 'e', 'f']}), 'test_data_1.csv', '/tmp'),
    (pd.DataFrame({'col1': [7, 8, 9], 'col2': ['g', 'h', 'i']}), 'test_data_2.csv', '/tmp')])
def test_save_to_csv(data, file_name, folder_path):
    folder_path = Path(folder_path)
    file_path = folder_path / file_name
    DataPoint(api_key).save_to_csv(data, file_name, folder_path)
    assert file_path.is_file()
    data_read = pd.read_csv(file_path)
    assert data.equals(data_read)


def test_save_to_csv_error(capsys):
    data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
    file_name = 'test_data'
    invalid_folder_path = '/this/is/an/invalid/path'
    data_point_instance = DataPoint(api_key)
    data_point_instance.save_to_csv(data, file_name, invalid_folder_path)
    captured = capsys.readouterr()
    assert "Error saving data to CSV: [Errno 30] Read-only file system: '/this'\n"'Check folder path exists & file name ends in ".csv"' in captured.out


@pytest.fixture
def sample_forecast_data():
    data = pd.DataFrame({'D': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                   24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
                             'F': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                   24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
                             'G': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                   24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
                             'H': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                   24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
                             'Pp': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                    24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
                             'S': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                   24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
                             'T': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                   24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
                             'V': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                   24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
                             'W': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                   24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
                             'U': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                   24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
                             '$': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                   24, 25, 26, 27, 28, 29, 30, 31, 32, 33]})
    data['W'] = data['W'].map(weather_dict)
    return data

def test_fix_columns(sample_forecast_data):
    input_df = sample_forecast_data
    output_df = DataPoint(api_key).fix_columns(input_df)
    expected_columns = ['Wind Direction', 'Feels Like Temperature',
                        'Wind Gust', 'Screen Relative Humidity', 'Precipitation Probability',
                        'Wind Speed', 'Temperature', 'Visibility', 'Weather Type',
                        'Max UV Index']
    assert output_df.columns.tolist() == expected_columns


def test_fix_columns_2(sample_forecast_data, capsys):
    input_df = sample_forecast_data
    del input_df['W']
    DataPoint(api_key).fix_columns(input_df)
    captured = capsys.readouterr()
    assert "Error mapping weather types: 'Weather Type'"'\nData being used may be incompatible with "weather_mapping" template' in captured.out


def test_fix_columns_3(sample_forecast_data):
    input_df = sample_forecast_data
    del input_df['$']
    output_df = DataPoint(api_key).fix_columns(input_df)
    expected_columns = ['Wind Direction', 'Feels Like Temperature',
                        'Wind Gust', 'Screen Relative Humidity', 'Precipitation Probability',
                        'Wind Speed', 'Temperature', 'Visibility', 'Weather Type',
                        'Max UV Index']
    assert output_df.columns.tolist() == expected_columns


@pytest.fixture
def weather_classifier():
    location_ids = ['3002', '353595']
    return WeatherClassifier(api_key, location_ids)


def test_format_data(weather_classifier):
    weather_classifier.format_data()
    assert weather_classifier.X_train is not None
    assert weather_classifier.X_test is not None
    assert weather_classifier.y_train is not None
    assert weather_classifier.y_test is not None


def test_format_missing_weather_type(weather_classifier, capsys):
    del weather_classifier.df['Weather Type']
    weather_classifier.format_data()
    captured = capsys.readouterr()
    assert "Dataframe does not contain a Weather Type column" in captured.out


def test_format_numeric_columns(weather_classifier, capsys):
    weather_classifier.df = weather_classifier.df.select_dtypes(include=['number'])
    weather_classifier.df['Weather Type'] = weather_classifier
    weather_classifier.format_data()
    captured = capsys.readouterr()
    assert "Dataframe does not contain non-numeric columns" in captured.out


@pytest.mark.parametrize('classifier, name', [
    (LogisticRegression, 'Logistic Regression'),
    (SGDClassifier, 'SGD Classifier')
                         ])
def test_train_and_print_report(classifier, name):
     classification_model = WeatherClassifier(api_key, ['3002']).train_and_print_report(classifier, name)
     assert isinstance(classification_model, pd.DataFrame)
     assert len(list(classification_model.values)) > 0


@pytest.mark.parametrize("classifier, name, outcome", [
    (LinearRegression, 'Linear Regression', None),
    (SGDRegressor, 'SGD Regressor', None)
                        ])
def test_false_train_report(classifier, name, outcome):
    report = WeatherClassifier(api_key, ['3002']).train_and_print_report(classifier, name)
    assert report == outcome
    assert report != pd.DataFrame
