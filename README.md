## Enrollments Dashboard

This project consists of an interactive dashboard that analyzes and visualizes enrollment data. It utilizes various machine learning techniques, including linear regression, decision tree regression, XGBoost, and H2O AutoML, to predict enrollment numbers based on different features.

### Data Preprocessing and Feature Engineering

The project starts by importing the necessary libraries and loading the enrollment data from two separate CSV files: `control_data.csv` and `experiment_data.csv`. These datasets are combined, missing data is removed, and additional features are engineered, such as the day of the week (DOW) and an experiment indicator. The data is then split into training and testing sets, and the categorical feature "DOW" is encoded into numerical values.

### Baseline Linear Regression

The project begins by establishing a baseline model using linear regression. The training set is used to fit the model, and predictions are made on the testing set. The metrics used to evaluate the model's performance include root mean squared error (RMSE), R-squared, and mean absolute error (MAE). Additionally, a plot comparing the true labels and predicted labels is generated.

### Decision Tree Regression

Next, a decision tree regression model is employed. The model's hyperparameters, such as maximum depth and minimum samples per leaf, are set, and the model is trained on the refined training set. Predictions are made on the testing set, and the same evaluation metrics are computed. A visualization of the true labels and predicted labels is also generated.

### XGBoost

The third model used is XGBoost, an implementation of gradient boosting. The parameters for the XGBoost model are defined, and the model is trained using the training set. Predictions are then made on the testing set, and the evaluation metrics are calculated. Finally, a plot comparing the true and predicted labels is generated.

### H2O AutoML

The last model utilized is H2O AutoML, which automates the process of training and evaluating multiple machine learning models. The training and testing sets are converted into H2O frames, and the features and target variable are specified. The AutoML process is initiated, and the best model obtained from the AutoML training is selected. Predictions are made on the testing set, and the evaluation metrics are calculated.

### Interactive Dashboard

The project concludes with the creation of an interactive dashboard using the Dash library. The dashboard includes components such as an H1 heading, original values plot, predicted values plot, and additional customizable elements. The original values plot visualizes the enrollment data by date, with different colors indicating the control and experimental groups. The predicted values plot displays the predicted enrollment numbers based on the selected value from a dropdown menu. For example, a line plot of the predicted enrollments can be displayed.

This project provides a comprehensive analysis of enrollment data and offers an interactive dashboard for visualizing and exploring the data using different machine learning models. It allows users to gain insights into enrollment trends and make predictions based on various factors.
