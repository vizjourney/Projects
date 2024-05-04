import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import sklearn

from data_processing import load_data, preprocess_data
from model_training import train_linear_regression, train_decision_tree, train_xgboost, train_h2o_automl, calculate_metrics, plot_preds


# Load and preprocess data
control_data, experiment_data = load_data("/Users/hydra/Desktop/Projects/AB_ML_test/data/control_data.csv", "/Users/hydra/Desktop/Projects/AB_ML_test/data/experiment_data.csv")
data_total = preprocess_data(control_data, experiment_data)

# Train models
linear_regression_model = train_linear_regression(data_total)
decision_tree_model = train_decision_tree(data_total)
xgboost_model = train_xgboost(data_total)
h2o_automl_model = train_h2o_automl(data_total)

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
    # Add code to generate predicted values based on selected model
    return fig

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
