#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 17:39:26 2023

@author: aliburak
"""

import os
import pandas as pd
import re

def extract_numeric_part(file_name):
    match = re.search(r'\d+', file_name)
    if match:
        return int(match.group())
    else:
        return 0

def reorganize_data(directory):
    data = {
        'consumer': [],
        'prosumer': []
    }

    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            print(f"Processing files from {subdir_path}")
            datasets = []

            # Sort the file names based on the numeric part
            file_names = sorted(os.listdir(subdir_path), key=extract_numeric_part)

            for file_name in file_names:
                if file_name.endswith('.csv'):
                    file_path = os.path.join(subdir_path, file_name)
                    df = pd.read_csv(file_path)
                    df = process_dataframe(df)
                    datasets.append(df)
            if subdir == 'consumer':
                data['consumer'] = datasets
            elif subdir == 'prosumer':
                data['prosumer'] = datasets
    
    return data

def process_dataframe(df):
    # Drop unnecessary columns
    df = df[['time', 'energy', 'energyOut']]
    
    # Convert time from Unix milliseconds to datetime range
    df['time'] = pd.to_datetime(df['time'], unit='ms', origin='unix', utc=True)
    df['time'] = df['time'].dt.tz_convert('CET')  # Replace 'YOUR_TIMEZONE' with your desired timezone
    
    # Calculate the first difference for energy consumption and production
    df['energy_diff'] = df['energy'].diff()* 10**-10
    df['energyOut_diff'] = df['energyOut'].diff()* 10**-10
    
    return df

# Example usage
directory_path = '/Users/aliburak/Desktop/Draft_DEDA/data'
organized_data = reorganize_data(directory_path)

# Accessing the organized data
consumer_data = organized_data['consumer']
prosumer_data = organized_data['prosumer']

# Example: Printing the first 10 rows of the first consumer dataset
print(consumer_data[0].head(10))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Specify the customer IDs to plot
customer_ids = [4,  7,  8, 20, 34, 41, 42, 50, 55, 62, 63, 95]

# Extract the corresponding datasets from the consumer_data list
datasets = [consumer_data[id - 1] for id in customer_ids]

# Plot the energy consumption for each consumer
for id, df in zip(customer_ids, datasets):
    # Resample the data to 15-minute intervals and calculate the sum
    resampled_df = df.resample('15T', on='time').sum()
    
    # Create a new figure and axis
    fig, ax = plt.subplots()
    
    # Plot the energy consumption
    ax.plot(resampled_df.index, resampled_df['energy_diff'])
    
    # Set the x-axis label
    ax.set_xlabel('Timestamp')
    
    # Set the y-axis label
    ax.set_ylabel('Energy Consumption (kWh)')
    
    # Set the title of the plot
    ax.set_title(f'Energy Consumption - Consumer {id}')
    
    # Format x-axis as months and years
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    # Rotate the x-axis labels for better readability
    fig.autofmt_xdate()
    
    # Save the plot as PNG
    plt.savefig('/Users/aliburak/Desktop/Draft_DEDA/graphs/consumer_graph_{id}.png'.format(id=id))
    
    # Display the plot
    plt.show()

# Specify the prosumer IDs to plot
prosumer_ids = [19, 24, 26, 30, 31, 72, 75, 83, 84, 85, 86, 89]

# Extract the corresponding datasets from the prosumer_data list
datasets = [prosumer_data[id - 1] for id in prosumer_ids]

# Plot the energy consumption for each prosumer
for id, df in zip(prosumer_ids, datasets):
    # Resample the data to 15-minute intervals and calculate the sum
    resampled_df = df.resample('15T', on='time').sum()
    
    # Create a new figure and axis
    fig, ax = plt.subplots()
    
    # Plot the energy consumption
    ax.plot(resampled_df.index, resampled_df['energy_diff'])
    
    # Set the x-axis label
    ax.set_xlabel('Timestamp')
    
    # Set the y-axis label
    ax.set_ylabel('Energy Consumption (kWh)')
    
    # Set the title of the plot
    ax.set_title(f'Energy Consumption - Prosumer {id}')
    
    # Format x-axis as months and years
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    # Rotate the x-axis labels for better readability
    fig.autofmt_xdate()
    
    # Save the plot as PNG
    plt.savefig('/Users/aliburak/Desktop/Draft_DEDA/graphs/prosumer_graph_{id}.png'.format(id=id))

    # Display the plot
    plt.show()



# Specify the prosumer IDs to plot
prosumer_ids = [19, 24, 26, 30, 31, 72, 75, 83, 84, 85, 86, 89]

# Extract the corresponding datasets from the prosumer_data list
datasets = [prosumer_data[id - 1] for id in prosumer_ids]

# Plot the energy consumption and production for each prosumer
for id, df in zip(prosumer_ids, datasets):
    # Resample the data to 15-minute intervals and calculate the sum
    resampled_df = df.resample('15T', on='time').sum()
    
    # Create a new figure and axis for each prosumer
    fig, ax = plt.subplots()
    
    # Plot the energy consumption
    ax.plot(resampled_df.index, resampled_df['energy_diff'], label='Consumption')
    
    # Plot the energy production
    ax.plot(resampled_df.index, resampled_df['energyOut_diff'], label='Production')
    
    # Set the x-axis label
    ax.set_xlabel('Timestamp')
    
    # Set the y-axis label
    ax.set_ylabel('Energy (kWh)')
    
    # Set the title of the plot
    ax.set_title(f'Energy Consumption and Production - Prosumer {id}')
    
    # Format x-axis as months and years
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    # Rotate the x-axis labels for better readability
    fig.autofmt_xdate()
    
    # Add a legend
    ax.legend()
   
    # Save the plot as PNG
    plt.savefig('/Users/aliburak/Desktop/Draft_DEDA/graphs/netprosumer_graph_{id}.png'.format(id=id))
    
    # Display the plot
    plt.show()
