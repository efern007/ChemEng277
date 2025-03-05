#Import Packages
import pandas as pd
import numpy as np

#Functions
def import_files(directory, file_prefix, no_files):
    """This function imports the csv files from a folder, and returns a list of them"""
    df_list = []
    for i in range(5,no_files):
        file_name = f"{file_prefix}{i:03}.csv"
        file_path = f"{directory}{file_name}"
        print(file_path)
        df_temp = pd.read_csv(file_path)
        df_list.append(df_temp)
    return df_list

def add_col_identifier(df_list, min_number):
   """This function adds a column with the test number to all data frames in a list"""
   id = min_number
   for df in df_list:
      df['cell_no'] = id
      id +=1
   return df_list

def convert_col_2_str(df_list, col_name):
   """This function converts a column to a string for all data frames in a list"""
   for df in df_list:
      df[col_name] = df[col_name].astype(str)
   return df_list

def concat_df(list_of_df, vertical_or_horizontal):
    """This function concats a list of dataframes either vertically or horizontally"""
    if vertical_or_horizontal == "horizontal":
        final_df = pd.concat(list_of_df, axis=1)
    
    if vertical_or_horizontal == "vertical":
        final_df = pd.concat(list_of_df, axis=0)
    return final_df

def calc_rest_per_cycle(df):
    """This function calculates the total seconds of resting the df battery experience
    per cycle, and returns a row with the rest sum, to fit the master table.
    
    Function to be called in calc_param_per_cycle()
    """
    filtered_df = df[df['State'] == 'R']
    max_cycle = df['Cyc#'].max()
    resting_row = np.zeros((1, max_cycle + 1))
    sum1 = 0
    for j in range(0,max_cycle):
        temp_df = filtered_df[filtered_df['Cyc#'] == j]
        if temp_df.empty:
            #if temp_df is empty assign 0 as sum_resting
            resting_row[0, j] = 0
        else:
            sum_resting = 0
            for i in range(1,len(temp_df)):
                if temp_df['Step (Sec)'].iloc[i] < temp_df['Step (Sec)'].iloc[i-1]:
                    sum_resting += temp_df['Step (Sec)'].iloc[i-1]
                i += 1          
            sum_resting = sum_resting + temp_df['Step (Sec)'].iloc[len(temp_df) - 1]
            resting_row[0, j] = sum_resting
            sum1 += j
    return resting_row

def calc_param_per_cycle(df):
    """This function creates a numpy array where the columns.size = number of cycles,
    and each row represents:
    row_0 = header (number of cycles)
    row_1 = mean of c rate per cycle
    row_2 = max of c rate per cycle
    row_3 = sum of resting time per cycle
    Other metrics can be easily added to this function using the line:
        result = np.vstack
    **Currently takes 2.9s to run per battery/cell**
    """
    mean_cycle = df.groupby('Cyc#')['Normalized Current (C-rate)'].mean()
    max_cycle = df.groupby('Cyc#')['Normalized Current (C-rate)'].max()
    header   = mean_cycle.index.to_numpy()
    mean_row = mean_cycle.to_numpy()
    max_row = max_cycle.to_numpy()
    resting_row = calc_rest_per_cycle(df)
    result = np.vstack((header, mean_row, max_row, resting_row))
    return result


def create_3d_array(df_list, pad_value=np.nan):
    """Creates a 3D array with all batteries, header (number of cycles), 
    mean of c rate per cycle, max of c rate per cycle and sum of resting time per 
    cycle"""
    list_2d = [calc_param_per_cycle(df) for df in df_list]
    max_cols = max(arr.shape[1] for arr in list_2d)
    pad_list = []
    for arr in list_2d:
        rows, cols = arr.shape
        if cols < max_cols:
            pad_width = ((0, 0), (0, max_cols - cols))
            arr = np.pad(arr, pad_width, mode='constant', constant_values=pad_value)
        pad_list.append(arr)
    master_array = np.stack(pad_list, axis=0)
    return master_array

def main():
    raw_list = import_files("ChemEng277/Data/RawData/", "Publishing_data_raw_data_cell", 96)
    aging_list = import_files("ChemEng277/Data/AgingData/", "Publishing_data_aging_data_cell", 96)
    raw_list = convert_col_2_str(raw_list, "State")
    raw_list = add_col_identifier(raw_list, 3)
    aging_list = add_col_identifier(aging_list, 3)
    raw_df = concat_df(raw_list, "vertical") #Provides a df with raw data from all cells
    aging_df = concat_df(aging_list, "vertical") #Provides a df with aging data from all cells
    
    """#Provides a 3D array with all batteries, header (number of cycles), 
    mean of c rate per cycle, max of c rate per cycle and sum of resting time per 
    cycle"""
    raw_3d = create_3d_array(raw_list) 
