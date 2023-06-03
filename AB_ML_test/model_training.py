import numpy as np
from sklearn.preprocessing import LabelEncoder
import sklearn.utils
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import h2o
from h2o.automl import H2OAutoML


def train_linear_regression(data):
    """
    Train a linear regression model.

    Args:
        data (pd.DataFrame): Preprocessed data.

    Returns:
        model: Trained linear regression model.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data.loc[:, data.columns != 'Enrollments'],
        data['Enrollments'],
        test_size=0.2,
        random_state=7
    )

    X_train_refined = X_train.drop(columns=['row_id'], axis=1)
    linear_regression = sm.OLS(y_train, X_train_refined).fit()
    return linear_regression


def train_decision_tree(data):
    """
    Train a decision tree model.

    Args:
        data (pd.DataFrame): Preprocessed data.

    Returns:
        model: Trained decision tree model.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data.loc[:, data.columns != 'Enrollments'],
        data['Enrollments'],
        test_size=0.2,
        random_state=7
    )

    X_train_refined = X_train.drop(columns=['row_id'], axis=1)
    dtree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=4, random_state=7)
    dtree.fit(X_train_refined, y_train)
    return dtree


def train_xgboost(data):
    """
    Train an XGBoost model.

    Args:
        data (pd.DataFrame): Preprocessed data.

    Returns:
        model: Trained XGBoost model.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data.loc[:, data.columns != 'Enrollments'],
        data['Enrollments'],
        test_size=0.2,
        random_state=7
    )

    lb = LabelEncoder()
    X_train['DOW'] = lb.fit_transform(X_train['DOW'])
    X_test['DOW'] = lb.transform(X_test['DOW'])

    DM_train = xgb.DMatrix(data=X_train_refined, label=y_train)
    DM_test = xgb.DMatrix(data=X_test_refined, label=y_test)
    parameters = {
        'max_depth': 6,
        'objective': 'reg:linear',
        'booster': 'gblinear',
        'n_estimators': 1000,
        'learning_rate': 0.2,
        'gamma': 0.01,
        'random_state': 7,
        'subsample': 1.
    }
    xg_reg = xgb.train(params=parameters, dtrain=DM_train, num_boost_round=8)
    return xg_reg


def train_h2o_automl(data):
    """
    Train an H2O AutoML model.

    Args:
        data (pd.DataFrame): Preprocessed data.

    Returns:
        model: Trained H2O AutoML model.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data.loc[:, data.columns != 'Enrollments'],
        data['Enrollments'],
        test_size=0.2,
        random_state=7
    )

    X_train['Enrollments'] = y_train
    X_test['Enrollments'] = y_test
    X_train_h2o = h2o.H2OFrame(X_train)
    X_test_h2o = h2o.H2OFrame(X_test)
    features = X_train.columns.values.tolist()
    target = "Enrollments"
    auto_h2o = H2OAutoML()
    auto_h2o.train(x=features, y=target, training_frame=X_train_h2o)
    auto_h2o = auto_h2o.leader
    return auto_h2o


def calculate_metrics(y_test, y_preds):
    """
    Calculate evaluation metrics.

    Args:
        y_test (pd.Series): Actual values.
        y_preds (pd.Series): Predicted values.

    Returns:
        None
    """
    rmse = np.sqrt(mean_squared_error(y_test, y_preds))
    r_sq = r2_score(y_test, y_preds)
    mae = mean_absolute_error(y_test, y_preds)

    print('RMSE Score: {}'.format(rmse))
    print('R2_Squared: {}'.format(r_sq))
    print('MAE Score: {}'.format(mae))


def plot_preds(y_test, y_preds, model_name):
    """
    Plot the predicted vs. actual values.

    Args:
        y_test (pd.Series): Actual values.
        y_preds (pd.Series): Predicted values.
        model_name (str): Name of the model.

    Returns:
        None
    """
    N = len(y_test)
    plt.figure(figsize=(10,5))
    original = plt.scatter(np.arange(1, N+1), y_test, c='blue')
    prediction = plt.scatter(np.arange(1, N+1), y_preds, c='red')
    plt.xticks(np.arange(1, N+1))
    plt.xlabel('# Observation')
    plt.ylabel('Enrollments')
    title = 'True labels vs. Predicted Labels ({})'.format(model_name)
    plt.title(title)
    plt.legend((original, prediction), ('Original', 'Prediction'))
    plt.show()
