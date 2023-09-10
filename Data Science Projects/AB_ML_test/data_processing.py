import os
import pandas as pd
import numpy as np
import sklearn


def load_data(control_filename, experiment_filename):
    """
    Load control and experiment data from CSV files.

    Args:
        control_filename (str): Filename of the control data CSV file.
        experiment_filename (str): Filename of the experiment data CSV file.

    Returns:
        control_data (pd.DataFrame): Control data as a Pandas DataFrame.
        experiment_data (pd.DataFrame): Experiment data as a Pandas DataFrame.
    """
    control_filepath = os.path.join("data", control_filename)
    experiment_filepath = os.path.join("data", experiment_filename)

    control_data = pd.read_csv(control_filepath)
    experiment_data = pd.read_csv(experiment_filepath)

    return control_data, experiment_data


def preprocess_data(control_data, experiment_data):
    """
    Preprocess the control and experiment data.

    Args:
        control_data (pd.DataFrame): Control data as a Pandas DataFrame.
        experiment_data (pd.DataFrame): Experiment data as a Pandas DataFrame.

    Returns:
        data_total (pd.DataFrame): Combined and preprocessed data.
    """
    # Combine control and experiment data
    data_total = pd.concat([control_data, experiment_data])

    # Remove missing data
    data_total.dropna(inplace=True)

    # Feature engineering
    data_total['row_id'] = data_total.index
    data_total['DOW'] = data_total['Date'].str.slice(start=0, stop=3)
    data_total['Experiment'] = np.random.randint(2, size=len(data_total))
    del data_total['Date'], data_total['Payments']
    data_total = sklearn.utils.shuffle(data_total)
    data_total = data_total[['row_id', 'Experiment', 'Pageviews', 'Clicks', 'DOW', 'Enrollments']]

    return data_total
