import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Load the dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# Display the first few rows of the dataset
print(df.head())

# Check data types
print(df.dtypes)

# Check for missing values
print(df.isnull().sum())


# Fill missing values with the mean of the column
df.fillna(df.mean(), inplace=True)

# Alternatively, drop rows with missing values
# df.dropna(inplace=True)

# Verify that there are no more missing values
print(df.isnull().sum())


# Summary statistics
print(df.describe())

# Group data by species and calculate mean
grouped_df = df.groupby('species').mean()

# Compute basic statistics of the numerical columns
print(df.describe())

# Group by species and compute the mean of each numerical column
group_means = df.groupby('species').mean()
print(group_means)

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Line chart example
# Using dummy data for demonstration
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='sepal_length', y='sepal_width')
plt.title('Line Chart of Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

# Bar Chart
plt.figure(figsize=(10, 6))
sns.barplot(x='species', y='petal_length', data=df)
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length')
plt.show()

# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['sepal_length'], bins=20, kde=True)
plt.title('Histogram of Sepal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.show()

#Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='sepal_length', y='petal_length', hue='species')
plt.title('Scatter Plot of Sepal Length vs Petal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.legend(title='Species')
plt.show()

# Error handling
try:
    df = pd.read_csv(url)
except FileNotFoundError:
    print("The file was not found.")
except pd.errors.EmptyDataError:
    print("No data found in the file.")
except Exception as e:
    print(f"An error occurred: {e}")
