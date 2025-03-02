import pandas as pd
def import_files(directory, file_prefix, no_files):
    
    df_list = []
    for i in range(3,no_files):
        file_name = f"{file_prefix}{i:03}.csv"
        file_path = f"{directory}{file_name}"
        df_temp = pd.read_csv(file_path)
        df_list[file_name] = df_temp
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

def main():
    raw_list = import_files("ChemEng277/Data/RawData/", "Publishing_data_raw_data_cell", 96)
    aging_list = import_files("ChemEng277/Data/AgingData/", "Publishing_data_aging_data_cell", 96)
    raw_list = convert_col_2_str(raw_list, "State")
    raw_list = add_col_identifier(raw_list, 3)
    aging_list = add_col_identifier(aging_list, 3)
    raw_df = concat_df(raw_list, "vertical")
    aging_df = concat_df(aging_list, "vertical")
