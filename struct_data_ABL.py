import os
import pandas as pd
import numpy as np
import pickle

def files_align(directory1, directory2):
    """This function checks if the files in two directories have the same final 3 
    character suffixes. Note: all files in the directories must be .csv files."""
    files1 = [f for f in os.listdir(directory1) if not f.startswith('.')]
    # print(files1)
    files2 = [f for f in os.listdir(directory2) if not f.startswith('.')]
    # print(files2)  
    suffixes1 = [file[-7:-4] for file in files1]
    suffixes2 = [file[-7:-4] for file in files2]
    # print(suffixes1)
    # print(suffixes2)
    suffixes1.sort()
    suffixes2.sort()
    # print(suffixes1)
    # print(suffixes2)
    if suffixes1 == suffixes2:
        print("The files in the two directories have the same suffixes.")
    else:
        raise ValueError("The files in the two directories do not have the same suffixes.")
    return

def import_files(directory, file_prefix, no_files, nrows=None):
    """This function imports the csv files from a folder, and returns a list of them"""
    df_list = []
    for i in range(0, no_files + 1):
        file_name = f"{file_prefix}{i:03}.csv"
        file_path = os.path.join(directory, file_name)
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist. Skipping.")
            continue
        # Specify the data types for each column
        dtype_dict = {
            'Test (Sec)': 'float64',
            'Step (Sec)': 'float64',
            'Normalized Current (C-rate)': 'float64',
            'Volts': 'float64',
            'Cyc#': 'int64',
            'Step': 'int64',
            'State': 'string',
            'Full Step #': 'string',
            'Loop3': 'int64',
            'Normalized Capacity (nominal capacity unit)': 'float64',
            'Normalized Charge Capacity [-]': 'float64',
            'Normalized Discharge Capacity [-]': 'float64',
            'Normalized Cumulative Capacity [-]': 'float64',
        }
        print(file_path)
        # Read the file with the specified number of rows
        df_temp = pd.read_csv(file_path, dtype=dtype_dict, low_memory=False, nrows=nrows)
        df_list.append(df_temp)
    print("Raw data imported.")
    # print structure of first dataframe
    print(df_list[0].head())
    # print structure of dataframes
    # print(df_list[0].info())
    # print structure of df_list
    print(f"Number of files in dataframe = {len(df_list)}.")
    """
    Note that there are only 92 files as the print statement shows above. 
    This is because the following files are missing: 000, 001, 002, 018, 083.

    This will be accounted for later in the code when separating the dataframes 
    into the four recipe types.
    """
    return df_list

def compress_raw_data(raw_list):
    """This function compresses the raw data at each cycle and enters the results as new columns in the processed data"""  
    df_list = []
    # raw list and processed list have the same number of dataframes
    for i in range(len(raw_list)):
        print(i)
        raw_init = raw_list[i]
        # only keep rows where State is 'R', 'C', or 'D'
        raw = raw_init[raw_init['State'].isin(['R', 'C', 'D'])].copy()
        # concatenate "Cyc#", "Full Step #" and "Loop3" to create a new column "Full Step #"
        raw.loc[:, 'Full Step #'] = raw['Cyc#'].astype(str) + "--" + raw['Full Step #'] + "--" + raw['Loop3'].astype(str)
        # calculate the max Step (hrs) for each Full Step #
        max_step_time = raw.groupby('Full Step #')['Step (Sec)'].max() / 3600
        # calculate the max Normalized Capacity (nominal capacity unit) for each Full Step #
        max_capacity = raw.groupby('Full Step #')['Normalized Capacity (nominal capacity unit)'].max()
        # calculate the average Normalized Current (C-rate) for each Full Step #
        avg_current = max_capacity / max_step_time
        # index the cycle number for each Full Step # (only one unique cycle number for each Full Step #)
        cycle_number = raw.groupby('Full Step #')['Cyc#'].max()
        # index the State for each Full Step # (only one unique state for each Full Step #)
        max_voltage = raw.groupby('Full Step #')['Volts'].max()
        state = raw.groupby('Full Step #')['State'].max()
        df_temp = pd.DataFrame({'Full Step #': max_step_time.index, 
                                'Max Step (hrs)': max_step_time.values, 
                                'Max Capacity': max_capacity.values, 
                                'Avg Current': avg_current.values, 
                                'Cycle': cycle_number.values, 
                                'State': state.values,
                                'Max Voltage': max_voltage})
        df_list.append(df_temp)
    # print the structure of the processed dataframe
    print(df_list[0].head())
    print(f'Processed Data has {len(df_list[0])} entries.')
    return df_list

def compress_to_cycle(processed_list, aging_list):
    """This function compresses the processed data to the cycle level and merges the aging data"""
    # create a list to store the processed dataframes
    df_list = []
    # processed list and aging list have the same number of dataframes
    for i in range(len(processed_list)):
        print(f'Index = {i}')
        processed = processed_list[i]
        aging = aging_list[i]
        # make various calculations on the processed data at each cycle
        # calculate the total time for each cycle
        total_time = processed.groupby('Cycle')['Max Step (hrs)'].sum()
        # Replace zero or missing values in total_time with a small positive number
        total_time = total_time.replace(0, np.nan).fillna(1e-10)
        # calculate the rest time 'R' for each cycle
        rest_time = processed[processed['State'] == 'R'].groupby('Cycle')['Max Step (hrs)'].sum()
        # Ensure all Series have the same index
        all_cycles = total_time.index
        rest_time = rest_time.reindex(all_cycles, fill_value=0)
        # calculate the fraction of rest time for each cycle
        rest_fraction = rest_time / total_time
        # calculate the charge time 'C' for each cycle
        charge_time = processed[processed['State'] == 'C'].groupby('Cycle')['Max Step (hrs)'].sum()
        charge_time = charge_time.reindex(all_cycles, fill_value=0)
        # calculate the fraction of charge time for each cycle
        charge_fraction = charge_time / total_time
        # calculate the discharge time 'D' for each cycle
        discharge_time = processed[processed['State'] == 'D'].groupby('Cycle')['Max Step (hrs)'].sum()
        discharge_time = discharge_time.reindex(all_cycles, fill_value=0)
        # calculate the fraction of discharge time for each cycle
        discharge_fraction = discharge_time / total_time
        # calculate the total capacity for each cycle
        total_capacity = processed.groupby('Cycle')['Max Capacity'].sum()
        total_capacity = total_capacity.reindex(all_cycles, fill_value=0)
        # calculate the average current for each cycle
        avg_current = total_capacity / total_time
        # calculate the max voltage for each cycle
        max_voltage = processed.groupby('Cycle')['Max Voltage'].max()
        max_voltage = max_voltage.reindex(all_cycles, fill_value=0)

        # create a dataframe to store the processed data
        df_temp1 = pd.DataFrame({'Cycle': total_time.index,
                                'Rest Fraction': rest_fraction.values,
                                'Charge Fraction': charge_fraction.values,
                                'Discharge Fraction': discharge_fraction.values,
                                'Total Capacity': total_capacity.values,
                                'Avg Current': avg_current.values,
                                'Max Voltage': max_voltage.values})
        # remove cycle 0 from the processed data
        df_temp1 = df_temp1[df_temp1['Cycle'] != 0]
        print(df_temp1.head())
        print(f'Processed Data has {len(df_temp1)} entries.')
        print(f'Aging data initially has {len(aging)} entries.')
        # remove rows in the aging data that are not in the processed data
        aging = aging[aging['Cycle'].isin(df_temp1['Cycle'])]
        print(f'After removing excess rows, there are {len(aging)} rows of aging data.')

        # merge the aging data with the processed data
        df_temp2 = pd.merge(aging, df_temp1, on='Cycle', how='left')
        df_list.append(df_temp2)
    # print the structure of the processed dataframe
    print(df_list[0].head())
    return df_list

def separate_dataframes(df_list):
    """This function separates the dataframes into the four recipe types
    and returns the four separated dataframes. Note that because runs 000, 
    001, 002, 018, and 083 are missing, the dataframes indexes will be off by
    a corresponding amount. Note: from Publishing_data_protocol_mapping_dic.json,
    the recipe types are as follows:
    Constant Current: run 3-24, 55, 56 => index 0-20, 51, 52
    Periodic: run 25-54 => index 21-50
    Synthetic: run 57-88 => index 53-83
    Real Driving: run 89-96 => index 84-91"""
    # initialize the four new arrays
    df_constant_current = {}
    df_periodic = {}
    df_synthetic = {}
    df_real_driving = {}
    count1, count2, count3, count4 = 0, 0, 0, 0
    # iterate through the list of dataframes
    for i, df in enumerate(df_list):
        print(f'Index = {i}')
        # check if the dataframe is empty
        if df.empty:
            print(f'Dataframe {i} is empty. Skipping.')
            continue
        print(f'Length of Dataframe {i} = {len(df)}')
        # check if the dataframe is missing
        if i < 21 or i == 51 or i == 52:
            # add the dataframe to the constant current dataframe
            df_constant_current[f'{count1}'] = df
            count1 += 1
        elif i < 51:
            # add the dataframe to the periodic dataframe
            df_periodic[f'{count2}'] = df
            count2 += 1
        elif i < 84:
            # add the dataframe to the synthetic dataframe
            df_synthetic[f'{count3}'] = df
            count3 += 1
        else:
            # add the dataframe to the real driving dataframe
            df_real_driving[f'{count4}'] = df
            count4 += 1
    # print the structure of the four dataframes
    print(f'Constant Current Dataframes: {df_constant_current.keys()}')
    print(f'Periodic Dataframes: {df_periodic.keys()}')
    print(f'Synthetic Dataframes: {df_synthetic.keys()}')
    print(f'Real Driving Dataframes: {df_real_driving.keys()}')
    print(f'Constant Current Dataframes Length: {count1}')
    print(f'Periodic Dataframes Length: {count2}')
    print(f'Synthetic Dataframes Length: {count3}')
    print(f'Real Driving Dataframes Length: {count4}')
    return df_constant_current, df_periodic, df_synthetic, df_real_driving

def main():
    # confirm the files in Data/RawData and Data/AgingData are from the same experiments
    files_align("Data/RawData", "Data/AgingData")
    # Raw Processing
    directory_raw = "Data/RawData/"
    file_prefix_raw = "Publishing_data_raw_data_cell_"
    no_files = 96
    # NOTE: import raw data with first 10,000 rows for testing, run with nrows=None for full data
    raw_list = import_files(directory_raw, file_prefix_raw, no_files, nrows=None)
    # Aging Processing
    directory_aging = "Data/AgingData/"
    file_prefix_aging = "Publishing_data_aging_summary_cell_"
    aging_list = import_files(directory_aging, file_prefix_aging, no_files, nrows=None) # import aging data with first 10,000 rows for testing
    # Make calculations on the raw data at each cycle and enter the results as new columns in the processed data
    init_processed_list = compress_raw_data(raw_list)
    # Compress the processed data to the cycle level and merge the aging data
    processed_list = compress_to_cycle(init_processed_list, aging_list)
    print(f'Processed List: {processed_list[0].head()}')
    print(f'Processed List Length: {len(processed_list[0])}')
    # Separate the dataframes into the four recipe types
    df_constant_current, df_periodic, df_synthetic, df_real_driving = separate_dataframes(processed_list)
    # Save the dataframes to a pickle file. If file already exists, it will be overwritten.
    with open('Data/ProcessedData/df_constant_current.pkl', 'wb') as f:
        pickle.dump(df_constant_current, f)
    with open('Data/ProcessedData/df_periodic.pkl', 'wb') as f:
        pickle.dump(df_periodic, f)
    with open('Data/ProcessedData/df_synthetic.pkl', 'wb') as f:
        pickle.dump(df_synthetic, f)
    with open('Data/ProcessedData/df_real_driving.pkl', 'wb') as f:
        pickle.dump(df_real_driving, f)
    return
   
if __name__ == "__main__":
    main()