# Data Analysis of COVID-19 Dataset
# Using pandas and matplotlib

import pandas as pd
import matplotlib.pyplot as plt

## 
# Load the dataset
df = pd.read_csv('covid_19_data.csv')

# Display first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Explore dataset structure
print("\nDataset info:")
print(df.info())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Clean the dataset - fill missing Province/State with 'Unknown' and drop rows with missing critical values
df['Province/State'].fillna('Unknown', inplace=True)
df.dropna(subset=['Confirmed', 'Deaths', 'Recovered'], inplace=True)

## Task 2: Basic Data Analysis
# Basic statistics
print("\nBasic statistics for numerical columns:")
print(df[['Confirmed', 'Deaths', 'Recovered']].describe())

# Convert ObservationDate to datetime format
df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])

# Group by Country and get mean values
country_stats = df.groupby('Country/Region')[['Confirmed', 'Deaths', 'Recovered']].mean()
print("\nAverage cases by country:")
print(country_stats.sort_values('Confirmed', ascending=False).head(10))

# Find interesting patterns
print("\nInteresting findings:")
print(f"1. Highest average confirmed cases: {country_stats['Confirmed'].idxmax()} with {country_stats['Confirmed'].max():.1f} cases")
print(f"2. Global death rate: {(df['Deaths'].sum()/df['Confirmed'].sum())*100:.2f}%")

## Task 3: Data Visualization
plt.figure(figsize=(15, 10))

# 1. Line chart - Daily global confirmed cases
plt.subplot(2, 2, 1)
daily_cases = df.groupby('ObservationDate')['Confirmed'].sum()
daily_cases.plot(kind='line', color='blue')
plt.title('Daily Global Confirmed COVID-19 Cases')
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.grid(True)

# 2. Bar chart - Top 10 countries by confirmed cases
plt.subplot(2, 2, 2)
top_countries = df.groupby('Country/Region')['Confirmed'].sum().nlargest(10)
top_countries.plot(kind='bar', color='orange')
plt.title('Top 10 Countries by Total Confirmed Cases')
plt.xlabel('Country')
plt.ylabel('Total Cases')
plt.xticks(rotation=45)

# 3. Histogram - Distribution of confirmed cases
plt.subplot(2, 2, 3)
plt.hist(df['Confirmed'], bins=50, color='green', edgecolor='black')
plt.title('Distribution of Confirmed Cases per Report')
plt.xlabel('Number of Confirmed Cases')
plt.ylabel('Frequency')
plt.yscale('log')  # Using log scale for better visualization

# 4. Scatter plot - Deaths vs Confirmed cases
plt.subplot(2, 2, 4)
plt.scatter(df['Confirmed'], df['Deaths'], alpha=0.5, color='red')
plt.title('Relationship Between Confirmed Cases and Deaths')
plt.xlabel('Confirmed Cases')
plt.ylabel('Deaths')
plt.xscale('log')
plt.yscale('log')

plt.tight_layout()
plt.show()

# Additional visualization - Recovery rate over time
plt.figure(figsize=(10, 6))
recovery_rate = df.groupby('ObservationDate').apply(
    lambda x: x['Recovered'].sum() / x['Confirmed'].sum() * 100
)
recovery_rate.plot(kind='line', color='purple')
plt.title('Global COVID-19 Recovery Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Recovery Rate (%)')
plt.grid(True)
plt.show()

print("\nAnalysis complete. Key visualizations displayed.")




        
            
