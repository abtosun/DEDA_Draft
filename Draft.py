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
# print(consumer_data[3].head(100000))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Specify the customer IDs to plot
consumer_ids = [4,  7,  8, 20, 34, 41, 42, 50, 55, 62, 63, 95]

# Extract the corresponding datasets from the consumer_data list
datasets = [consumer_data[id - 1] for id in consumer_ids]

# Plot the energy consumption for each consumer
for id, df in zip(consumer_ids, datasets):
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
    # plt.savefig('/Users/aliburak/Desktop/Draft_DEDA/graphs/consumer_graph_{id}.png'.format(id=id))
    
    # Display the plot
    # plt.show()

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
    # plt.savefig('/Users/aliburak/Desktop/Draft_DEDA/graphs/prosumer_graph_{id}.png'.format(id=id))

    # Display the plot
    # plt.show()



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
    # plt.savefig('/Users/aliburak/Desktop/Draft_DEDA/graphs/netprosumer_graph_{id}.png'.format(id=id))
    
    # Display the plot
    # plt.show()
################################################

# Specify the consumer and prosumer IDs to include
consumer_ids = [id for id in range(1, 101) if id not in [13, 21, 26, 35, 46, 53, 57, 67, 76, 78, 80, 81]]
prosumer_ids = [19, 24, 26, 30, 72, 75, 83, 89]

# Initialize an empty DataFrame to store the aggregate data
aggregate_data = pd.DataFrame(columns=['Timestamp', 'Aggregate Production (Prosumers)', 'Aggregate Consumption (Prosumers)', 'Aggregate Consumption (Consumers)'])

# Calculate aggregate production for the specified prosumers
for prosumer_id in prosumer_ids:
    # Extract the corresponding prosumer dataset
    df = prosumer_data[prosumer_id - 1].copy()
    
    # Resample the data to 15-minute intervals and calculate the sum
    resampled_df = df.resample('15T', on='time').sum()
    
    # Assign the aggregate production values to the corresponding time intervals
    aggregate_data['Aggregate Production (Prosumers)'] = aggregate_data['Aggregate Production (Prosumers)'].add(resampled_df['energyOut_diff'], fill_value=0)

# Calculate aggregate consumption for the specified prosumers
for prosumer_id in prosumer_ids:
    # Extract the corresponding prosumer dataset
    df = prosumer_data[prosumer_id - 1].copy()
    
    # Resample the data to 15-minute intervals and calculate the sum
    resampled_df = df.resample('15T', on='time').sum()
    
    # Assign the aggregate consumption values to the corresponding time intervals
    aggregate_data['Aggregate Consumption (Prosumers)'] = aggregate_data['Aggregate Consumption (Prosumers)'].add(resampled_df['energy_diff'], fill_value=0)

# Calculate aggregate consumption for the specified consumers
for consumer_id in consumer_ids:
    # Extract the corresponding consumer dataset
    df = consumer_data[consumer_id - 1].copy()
    
    # Resample the data to 15-minute intervals and calculate the sum
    resampled_df = df.resample('15T', on='time').sum()
    
    # Assign the aggregate consumption values to the corresponding time intervals
    aggregate_data['Aggregate Consumption (Consumers)'] = aggregate_data['Aggregate Consumption (Consumers)'].add(resampled_df['energy_diff'], fill_value=0)

# Set the 'Timestamp' column to match the index
aggregate_data['Timestamp'] = aggregate_data.index

# Reset the index and reorder the columns
aggregate_data = aggregate_data.reset_index(drop=True)[['Timestamp', 'Aggregate Production (Prosumers)', 'Aggregate Consumption (Prosumers)', 'Aggregate Consumption (Consumers)']]

aggregate_data['Net Load'] = aggregate_data['Aggregate Consumption (Prosumers)'] + aggregate_data['Aggregate Consumption (Consumers)'] - aggregate_data['Aggregate Production (Prosumers)']

# Print the resulting aggregate data
print(aggregate_data)

###########################


size = [100, 200, 500, 1000, 2000, 1090]
net_load = aggregate_data['Net Load']
battery_data = pd.DataFrame(index=range(len(net_load)), columns=[f"Charge {s}" for s in size] + [f"SOC {s}" for s in size] + [f"Charge/Discharge {s}" for s in size])

for s in range(len(size)):
    column_charge = f"Charge {size[s]}"
    column_soc = f"SOC {size[s]}"
    column_charge_discharge = f"Charge/Discharge {size[s]}"
    
    beg = size[s] / 2  # assumption: beginning SOC of battery is 50%
    min_size = 0.05 * size[s]  # assumption: battery does not get discharged to less than 5% of total battery capacity
    max_size = 0.95 * size[s]  # assumption: battery does not get charged to more than 95% of total battery capacity
    
    battery_data[column_charge] = pd.NA
    battery_data[column_soc] = pd.NA
    battery_data[column_charge_discharge] = pd.NA
    
    for i in range(len(battery_data)):
        if i == 0:
            battery_data.at[i, column_charge] = beg - net_load[i]
        else:
            prev = battery_data.at[i - 1, column_charge]
            nl = net_load[i]
            battery_data.at[i, column_charge] = prev - nl
        
        battery_data.at[i, column_charge] = max(min_size, min(battery_data.at[i, column_charge], max_size))
        battery_data.at[i, column_soc] = battery_data.at[i, column_charge] / size[s]
        
        if i != 0:
            battery_data.at[i, column_charge_discharge] = battery_data.at[i, column_charge] - battery_data.at[i - 1, column_charge]
            
###########################

battery_data.insert(0, 'Timestamp', aggregate_data['Timestamp'])





# import matplotlib.pyplot as plt
# import numpy as np
# from plotnine import *

# size = [100, 200, 500, 1000, 2000, 1090]
# plotgrids = []

# for s in range(len(size)):
#     id = size[s]
#     column_soc = f"SOC {size[s]}"

#     t = battery_data[['Timestamp', column_soc]].copy()
#     t['Timestamp'] = pd.to_datetime(t['Timestamp'])  # Convert timestamp to datetime format
#     t['Timestamp'] = t['Timestamp'].dt.to_period('M')  # Convert timestamp to month format

#     p = (
#         ggplot(t, aes(x='Timestamp', y=column_soc))
#         + geom_line()
#         + scale_y_continuous(limits=[0, 1])  # Set the y-axis limits to 0 and 1
#         + labs(y="SOC")
#         + xlab("Month")
#     )

#     plotgrids.append(p)

# for i in range(len(plotgrids)):
#     g = plotgrids[i]
#     g.save(f"/Users/aliburak/Desktop/Draft_DEDA/battery_SOC_{size[i]}.png", dpi=300)
#     import numpy as np
# from plotnine import *

# size = 1090

# column_soc = f"SOC {size}"
# t = battery_data[['Timestamp', column_soc]].copy()
# t['Timestamp'] = pd.to_datetime(t['Timestamp'])  # Convert timestamp to datetime format
# t['Timestamp'] = t['Timestamp'].dt.to_period('M')  # Convert timestamp to month format

# p = (
#     ggplot(t, aes(x='Timestamp', y=column_soc))
#     + geom_line()
    
#     + labs(y="SOC")
#     + xlab("Month")
# )

# p.draw()

