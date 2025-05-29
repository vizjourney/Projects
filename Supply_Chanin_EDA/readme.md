# Supply Chain Data Analysis and Fraud Detection

## Overview

This project focuses on analyzing supply chain data to gain insights into various aspects of the supply chain and to identify potential fraudulent activities. It involves data preprocessing, extensive exploratory data analysis (EDA) with various visualizations, and the implementation of machine learning models for fraud detection.

## Data

The dataset used in this project is sourced from the "DataCoSupplyChainDataset.csv" file. It contains information on various attributes related to the supply chain, including order details, customer information, product details, and delivery status.

## Requirements

The following Python libraries are used in this project:

*   `numpy`: For numerical computations.
*   `pandas`: For data manipulation and analysis.
*   `seaborn`: For creating informative statistical graphics.
*   `matplotlib`: For creating static, interactive, and animated visualizations.
*   `plotly`: For creating interactive plots and visualizations.
*   `sklearn`: For machine learning tasks, including preprocessing, model selection, and model training.
*   `xgboost`: For implementing the XGBoost algorithm.
*   `scipy`: For scientific and technical computing.

## Analysis and Methodology

The project follows a structured approach, including:

1.  **Data Loading and Initial Inspection:** Loading the dataset and performing initial checks on its structure and content.
2.  **Data Preprocessing:** Handling missing values, encoding categorical features, and preparing the data for analysis and modeling.
3.  **Exploratory Data Analysis (EDA):**
    *   Analyzing delivery status trends.
    *   Investigating customer behavior and order patterns.
    *   Analyzing sales trends by product, category, region, and payment type.
    *   Visualizing geographical distribution of profit.
    *   Analyzing sales over time (yearly, quarterly, monthly).
4.  **Feature Selection:** Using correlation analysis and feature importance techniques (like f-regression) to select relevant features for the machine learning model.
5.  **Model Selection and Training:** Exploring different classification models (Logistic Regression, RandomForestClassifier, KNeighborsClassifier, GaussianNB, SGDClassifier, DecisionTreeClassifier, XGBClassifier) and selecting the best-performing model.
6.  **Model Evaluation:** Evaluating the selected model's performance using appropriate metrics.

## Visualizations

The notebook includes various visualizations created using `matplotlib`, `seaborn`, and `plotly` to illustrate key findings from the data. These visualizations cover aspects like:

*   Distribution of delivery statuses.
*   Number of orders by customer segment.
*   Sales by product category and payment type.
*   Geographical profit distribution.
*   Sales trends over time.

## Machine Learning Model

The project implements a machine learning model (specifically, a RandomForestClassifier with optimized hyperparameters) to identify potential instances of "SUSPECTED_FRAUD" based on the selected features. The code includes steps for:

*   Splitting the data into training and testing sets.
*   Creating pipelines for different classifiers.
*   Using `RandomizedSearchCV` to find the best hyperparameters for the RandomForestClassifier.
*   Evaluating the model's performance.

## How to Run the Code

1.  Clone the repository: `git clone [repository_url]`
2.  Install the required libraries: `pip install -r requirements.txt` (You'll need to create a `requirements.txt` file listing the libraries mentioned in the Requirements section).
3.  Run the Jupyter Notebook: `jupyter notebook Supply_Chain_EDA.ipynb` (Assuming the notebook is named `Supply_Chain_EDA.ipynb`)

## Files in the Repository

*   `Supply_Chain_EDA.ipynb`: The Jupyter Notebook containing the code for data analysis and fraud detection.
*   `DataCoSupplyChainDataset.csv`: The dataset used in the project.
*   `requirements.txt`: A file listing the required Python libraries.

## Future Enhancements

*   Explore other machine learning algorithms for fraud detection.
*   Implement more advanced feature engineering techniques.
*   Perform hyperparameter tuning on other models.
*   Develop a web application or dashboard to visualize the results and allow interactive exploration.
