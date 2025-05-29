# Dynamic Pricing Strategy for Ride Services

## Overview

This project explores and implements a dynamic pricing strategy for ride services. The goal is to optimize ride costs based on real-time factors such as the number of riders, the number of available drivers, the type of vehicle requested, and the expected duration of the ride. The project involves data analysis, visualization, and the development of a predictive model to determine the adjusted ride cost.

## Data

The project uses a dataset containing information related to ride services, including:

*   `Number_of_Riders`: The number of people requesting rides.
*   `Number_of_Drivers`: The number of available drivers.
*   `Location_Category`: The location of the ride request (Urban, Suburban, Rural).
*   `Customer_Loyalty_Status`: The loyalty status of the customer (Silver, Regular, etc.).
*   `Number_of_Past_Rides`: The customer's ride history.
*   `Average_Ratings`: The customer's average rating.
*   `Time_of_Booking`: The time of day the ride is booked.
*   `Vehicle_Type`: The type of vehicle requested (Premium, Economy).
*   `Expected_Ride_Duration`: The estimated duration of the ride.
*   `Historical_Cost_of_Ride`: The historical cost of a similar ride.

## Requirements

The following Python libraries are required to run this project:

*   `pandas`: For data manipulation and analysis.
*   `numpy`: For numerical operations.
*   `plotly`: For creating interactive visualizations.
*   `sklearn`: For machine learning tasks, including data splitting and model training.
*   `sklearn.ensemble.RandomForestRegressor`: The model used for price prediction.

## Methodology

The project follows these steps:

1.  **Data Loading and Exploration:** Load the dataset and perform initial data exploration to understand the data structure and key variables.
2.  **Data Preprocessing:** Clean and preprocess the data, including handling missing values and encoding categorical features.
3.  **Exploratory Data Analysis (EDA):** Visualize relationships between variables using scatter plots, box plots, and heatmaps to identify patterns and correlations.
4.  **Feature Engineering:** Calculate demand and supply multipliers based on rider and driver numbers to incorporate real-time market conditions into the pricing model.
5.  **Dynamic Pricing Calculation:** Implement a dynamic pricing formula to calculate the `adjusted_ride_cost` based on historical cost and the engineered features.
6.  **Profitability Analysis:** Analyze the profitability of rides under the dynamic pricing strategy compared to historical pricing.
7.  **Model Development:** Train a Random Forest Regressor model to predict the `adjusted_ride_cost` based on relevant features.
8.  **Model Evaluation:** Evaluate the performance of the trained model using visualizations like actual vs. predicted values plots.
9.  **Prediction Function:** Create a function to predict the ride cost based on user-provided input.

## Visualizations

The notebook includes several visualizations to illustrate the data and the impact of dynamic pricing, such as:

*   Scatter plot of Expected Ride Duration vs. Historical Cost of Ride with a trendline.
*   Box plot of Historical Cost of Ride Distribution by Vehicle Type.
*   Correlation Matrix heatmap.
*   Donut chart showing the profitability of rides with dynamic pricing.
*   Scatter plot of Actual vs. Predicted `adjusted_ride_cost` values.

## How to Run the Code

1.  Clone the repository: `git clone [repository_url]`
2.  Install the required packages: `pip install -r requirements.txt` (Create a `requirements.txt` file with the libraries listed in the Requirements section).
3.  Run the Jupyter Notebook: `jupyter notebook [notebook_name].ipynb` (Replace `[notebook_name].ipynb` with the actual name of your notebook file).

## Files in the Repository

*   `[notebook_name].ipynb`: The Jupyter Notebook containing the code for the dynamic pricing project.
*   `dynamic_pricing.csv`: The dataset used in the project.
*   `requirements.txt`: A file listing the required Python libraries.

## Future Enhancements

*   Explore other regression models for price prediction.
*   Incorporate more features into the model (e.g., time of day, weather conditions).
*   Develop a more sophisticated dynamic pricing algorithm.
*   Build a user interface to interact with the pricing model.
