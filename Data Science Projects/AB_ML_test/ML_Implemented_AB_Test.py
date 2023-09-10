#import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import sklearn.utils
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import h2o
from h2o.automl import H2OAutoML

# Load and preprocess data
control_data = pd.read_csv('control_data.csv')
experiment_data = pd.read_csv('experiment_data.csv')

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

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(
    data_total.loc[:, data_total.columns != 'Enrollments'],
    data_total['Enrollments'],
    test_size=0.2,
    random_state=7
)

# Converting strings to numbers
lb = LabelEncoder()
X_train['DOW'] = lb.fit_transform(X_train['DOW'])
X_test['DOW'] = lb.transform(X_test['DOW'])

# Helper functions

def calculate_metrics(y_test, y_preds):
    rmse = np.sqrt(mean_squared_error(y_test, y_preds))
    r_sq = r2_score(y_test, y_preds)
    mae = mean_absolute_error(y_test, y_preds)

    print('RMSE Score: {}'.format(rmse))
    print('R2_Squared: {}'.format(r_sq))
    print('MAE Score: {}'.format(mae))

def plot_preds(y_test, y_preds, model_name):
    N = len(y_test)
    plt.figure(figsize=(10,5))
    original = plt.scatter(np.arange(1, N+1), y_test, c='blue')
    prediction = plt.scatter(np.arange(1, N+1), y_preds, c='red')
    plt.xticks(np.arange(1, N+1))
    plt.xlabel('# Oberservation')
    plt.ylabel('Enrollments')
    title = 'True labels vs. Predicted Labels ({})'.format(model_name)
    plt.title(title)
    plt.legend((original, prediction), ('Original', 'Prediction'))
    plt.show()

# Linear regression: A baseline

X_train_refined = X_train.drop(columns=['row_id'], axis=1)
linear_regression = sm.OLS(y_train, X_train_refined).fit()
X_test_refined = X_test.drop(columns=['row_id'], axis=1)
y_preds = linear_regression.predict(X_test_refined)
calculate_metrics(y_test, y_preds)
plot_preds(y_test, y_preds, 'Linear Regression')

# Decision Tree

dtree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=4, random_state=7)
dtree.fit(X_train_refined, y_train)
y_preds = dtree.predict(X_test_refined)
calculate_metrics(y_test, y_preds)
plot_preds(y_test, y_preds, 'Decision Tree')

# XGBoost

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
y_preds = xg_reg.predict(DM_test)
calculate_metrics(y_test, y_preds)
plot_preds(y_test, y_preds, 'XGBoost')

# H2O AutoML

h2o.init()
X_train['Enrollments'] = y_train
X_test['Enrollments'] = y_test
X_train_h2o = h2o.H2OFrame(X_train)
X_test_h2o = h2o.H2OFrame(X_test)
features = X_train.columns.values.tolist()
target = "Enrollments"
auto_h2o = H2OAutoML()
auto_h2o.train(x=features, y=target, training_frame=X_train_h2o)
auto_h2o = auto_h2o.leader
X_test_temp = X_test.copy()
del X_test_temp['Enrollments']
X_test_h2o_copy = h2o.H2OFrame(X_test_temp)
y_preds = auto_h2o.predict(X_test_h2o_copy)
y_preds = h2o.as_list(y_preds["predict"])
calculate_metrics(y_test, y_preds)

# Create the Dash app
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("Enrollments Dashboard"),
    
    # Original values plot
    html.Div([
        dcc.Graph(id="original-plot")
    ]),
    
    # Predicted values plot
    html.Div([
        dcc.Graph(id="predicted-plot")
    ]),
    
    # Other components or layout elements can be added here
    
])

# Callback to update the original values plot
@app.callback(Output("original-plot", "figure"), [Input("my-dropdown", "value")])
def update_original_plot(selected_value):
    # Code to update the original values plot based on the selected value (if applicable)
    # Example: Create a bar plot of Pageviews by Date
    fig = px.bar(data_total, x='Date', y='Pageviews', color='Experiment')
    return fig

# Callback to update the predicted values plot
@app.callback(Output("predicted-plot", "figure"), [Input("my-dropdown", "value")])
def update_predicted_plot(selected_value):
    # Code to update the predicted values plot based on the selected value (if applicable)
    # Example: Create a line plot of Enrollments predictions
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_test.index, y=y_preds, mode='lines', name='Predicted Enrollments'))
    fig.update_layout(xaxis_title="Observation", yaxis_title="Enrollments")
    return fig

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)