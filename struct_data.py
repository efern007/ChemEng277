import os
import pandas as pd
import numpy as np

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
        # concatenate "Full Step #" and "Loop3" to create a new column "Full Step #"
        raw.loc[:, 'Full Step #'] = raw['Full Step #'] + "--" + raw['Loop3'].astype(str)
        # calculate the max Step (hrs) for each Full Step #
        max_step_time = raw.groupby('Full Step #')['Step (Sec)'].max() / 3600
        # calculate the max Normalized Capacity (nominal capacity unit) for each Full Step #
        max_capacity = raw.groupby('Full Step #')['Normalized Capacity (nominal capacity unit)'].max()
        # calculate the average Normalized Current (C-rate) for each Full Step #
        avg_current = max_capacity / max_step_time
        # index the cycle number for each Full Step # (only one unique cycle number for each Full Step #)
        cycle_number = raw.groupby('Full Step #')['Cyc#'].max()
        # index the State for each Full Step # (only one unique state for each Full Step #)
        state = raw.groupby('Full Step #')['State'].max()
        df_temp = pd.DataFrame({'Full Step #': max_step_time.index, 'Max Step (hrs)': max_step_time.values, 
                                'Max Capacity': max_capacity.values, 'Avg Current': avg_current.values, 
                                'Cycle Number': cycle_number.values, 'State': state.values})
        df_list.append(df_temp)
    # print the structure of the processed dataframe
    print(df_list[0].head())
    return df_list

def main():
    # confirm the files in Data/RawData and Data/AgingData are from the same experiments
    files_align("Data/RawData", "Data/AgingData")
    # Raw Processing
    directory_raw = "Data/RawData/"
    file_prefix_raw = "Publishing_data_raw_data_cell_"
    no_files = 96
    # NOTE: import raw data with first 10,000 rows for testing, run with nrows=None for full data
    raw_list = import_files(directory_raw, file_prefix_raw, no_files, nrows=10000)
    # Aging Processing
    directory_aging = "Data/AgingData/"
    file_prefix_aging = "Publishing_data_aging_summary_cell_"
    aging_list = import_files(directory_aging, file_prefix_aging, no_files, nrows=None) # import aging data with first 10,000 rows for testing
    # Make calculations on the raw data at each cycle and enter the results as new columns in the processed data
    init_processed_list = compress_raw_data(raw_list)
   
if __name__ == "__main__":
    main()