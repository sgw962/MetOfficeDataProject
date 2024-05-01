from src.data.data_upload import DataPoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import xlsxwriter
import os


class WeatherClassifier:
    """
    Initiate this class with a valid met office API - this allows access to a dataframe consisting of 3hourly, 5-day
    forecasts for 30 locations from met office data. It will also require the location IDs of the locations you want to
    analyse. The class will use this dataframe to predict weather type based on all other features using different
    classification models.
    """
    def __init__(self, api_key, location_ids):
        """
        Initialises the class establishing the df and 'format_data()' method/ components for use in subsequent methods.
        """
        self.api_key = api_key
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        #Above are all required in subsequent methods so established as self but need to be formed in format_data
        datapoint = DataPoint(api_key)
        self.location_ids = location_ids
        self.df = datapoint.fix_columns(datapoint.get_forecast_data(location_ids))
        #format_data established due to X/y test/train
        self.format_data()

    def format_data(self):
        """
        Formats the data for use on classification models. Uses LabelEncoder and
        train_test_split to produce 'X' and 'y' for 'train' and 'test' data to be used in the following method.
        """
        try:
            if 'Weather Type' not in self.df:
                raise ValueError('Dataframe does not contain a Weather Type column')
            else:
                X = self.df.drop('Weather Type', axis=1)
                y = self.df['Weather Type']

            if len(X.select_dtypes(include=['object']).columns) == 0:
                raise ValueError('Dataframe does not contain non-numeric columns')
            else:
                label_encoder = LabelEncoder()
                for column in X.select_dtypes(include=['object']).columns:
                    X[column] = label_encoder.fit_transform(X[column])
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25, random_state=0)
            return self.X_train, self.X_test, self.y_train, self.y_test
        except Exception as e:
            print(f'Error occurred while formatting data: {e}')
            print('Try checking the API key is valid')

    def train_and_print_report(self, classifier, name):
        """
        Call this function with a valid classification model and the name of the classifier used. The function then builds and
        runs the model on the data, returning accuracy metrics for the models performance. It also returns an Excel copy
        of the performance metrics table in the "reports" folder of this repository.
        """
        try:
            model = classifier(random_state=0)
            model.fit(self.X_train, self.y_train)

            predictions = model.predict(self.X_test)

            print(f'{name} Report:')
            print(classification_report(self.y_test, predictions, zero_division=np.nan))
            df_report = pd.DataFrame(classification_report(self.y_test, predictions, output_dict=True))
            file_path = os.path.join('/Users/seanwhite/OneDrive - University of Greenwich/Documents/Advanced '
                                         'Programming/Coursework/CourseworkReport&Code/reports/figures/',
                                         f'{name}_report.xlsx')
            df_report.to_excel(file_path, index=False)
            return df_report
        except Exception as e:
            print(f'Error running classifier: {e}')
            print(f'Check "{name}" is a valid classification model')
