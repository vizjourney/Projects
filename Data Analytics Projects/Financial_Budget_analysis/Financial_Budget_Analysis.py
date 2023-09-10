### Import Libraries and Load Data

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt  
##Load the data from the CSV file 
data = pd.read_csv("/Users/hydra/Desktop/Projects/Financial_Budget_analysis/India_budget_2021.csv")

### Data Exploration and Preprocessing

# Display the first few rows of the data 
data.head()

### Data Cleaning

# Drop rows with missing values 
data_cleaned = data.dropna()  
##Select specific rows 
indices_to_select = [0, 8, 11, 14, 18, 23, 41, 42, 43] 
data_selected = data_cleaned.iloc[indices_to_select]

### Additional Data Transformation

# Add a custom row 
custom_row = {'Department /Ministry': 'OTHERS', 'Fund allotted(in ₹crores)': 592971.08}
data_final = data_selected.append(custom_row, ignore_index=True)

### Visualization: Bar Plot

# Create a bar plot 
plt.figure(figsize=(10, 6)) 
plt.bar(data_final['Department /Ministry'], data_final['Fund allotted(in ₹crores)'])
plt.xlabel('Department / Ministry')
plt.ylabel('Fund allotted (in ₹crores)') 
plt.title('Budget Allocation for Different Departments / Ministries') 
plt.xticks(rotation=45, ha="right") 
plt.tight_layout()
plt.show()

###Visualization: Pie Chart

# Create a pie chart
df = data["Fund allotted(in ₹crores)"]
labels = data["Department /Ministry"]
plt.figure(figsize=(7,7))
plt.pie(df, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, shadow =True)
central_circle = plt.Circle((0, 0), 0.5, color='white')
fig = plt.gcf()
fig.gca().add_artist(central_circle)
plt.rc('font', size=12)
plt.title("Distribution of The Budget", fontsize=20)
plt.show()
