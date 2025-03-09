import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

"""As a note for data analysis: all models will be trained and validated individually for each of the three 
datasets: constant current, periodic, and synthetic. The real driving dataset will be used for testing purposes
only. Normalized Discharge Capacity (or State of Health - "SOH") will be the target (y) variable for all models.
Once random keys are determined for each dataset, those keys will be averaged to create one 2D list of parameters
versus cycle number. The same method will be used for the validation set. The target variable matrix will be
2D with SOH vs cycle number. These models will be fitted to Linear, Lasso, and Ridge Regression. Lasso and 
Ridge Regression will be tested for various alpha (lambda) hyperparameters [0.1, 0.25, 0.5, 0.75, 0.9]. 
For both the train and validation sets, the R^2 value will be calculated for each model. The model with the
highest R^2 value will be selected to proceed to the test set. Because of potential variability in the train/validation
split, a series of random combinations of train/validation sets will be tested. The model with the highest average
R^2 value will be selected for the test set. The test set will be individually tested against the model."""

# Load data (.pkl dictionary files) from Data/For_Analysis
def load_data():
    with open("Data/For_Analysis/df_constant_current.pkl", "rb") as f:
        data_constant_current = pickle.load(f)
    with open("Data/For_Analysis/df_periodic.pkl", "rb") as f:
        data_periodic = pickle.load(f)
    with open("Data/For_Analysis/df_synthetic.pkl", "rb") as f:
        data_synthetic = pickle.load(f)
    with open("Data/For_Analysis/df_real_driving.pkl", "rb") as f:
        data_real_driving = pickle.load(f)
    print("Data loaded.")
    return data_constant_current, data_periodic, data_synthetic, data_real_driving

def inspect_dictionary(d):
    print("Keys:", d.keys())
    # Print the first few rows of the first DataFrame in the dictionary
    print(f"DataFrame for {list(d.keys())[0]}:")
    print(d[list(d.keys())[0]].head())
    for key in d.keys():
        print(f"DataFrame for {key}:")
        # print(d[key].head())  # Print the first few rows of the DataFrame
        # print number of rows and columns
        print(f"Number of rows: {d[key].shape[0]}")
        print(f"Number of columns: {d[key].shape[1]}")
    return

def inspect_all_data(data_constant_current, data_periodic, data_synthetic, data_real_driving):
    print("Inspecting Constant Current Data:")
    inspect_dictionary(data_constant_current)
    print("Inspecting Periodic Data:")
    inspect_dictionary(data_periodic)
    print("Inspecting Synthetic Data:")
    inspect_dictionary(data_synthetic)
    print("Inspecting Real Driving Data:")
    inspect_dictionary(data_real_driving)
    return

def create_train_val_sets(data):
    # Save the headers of the DataFrames
    headers = data[list(data.keys())[0]].columns
    # Create a dictionary to store the train and validation sets
    train_set = {}
    val_set = {}
    for key in data.keys():
        # Randomly shuffle the data
        data[key] = data[key].sample(frac=1, random_state=1).reset_index(drop=True)
        # Split the data into train and validation sets
        train_set[key], val_set[key] = train_test_split(data[key], test_size=0.2, random_state=6)
    
    # Determine the maximum number of rows across all DataFrames
    max_rows = max(df.shape[0] for df in data.values())
    
    # Initialize sum and count arrays
    sum_train_set = np.zeros((max_rows, data[list(data.keys())[0]].shape[1]))
    count_train_set = np.zeros((max_rows, 1))
    sum_val_set = np.zeros((max_rows, data[list(data.keys())[0]].shape[1]))
    count_val_set = np.zeros((max_rows, 1))
    
    # Sum the train and validation sets and count the occurrences
    for key in data.keys():
        train_df = train_set[key].values
        val_df = val_set[key].values
        for i in range(train_df.shape[0]):
            sum_train_set[i] += train_df[i]
            count_train_set[i] += 1
        for i in range(val_df.shape[0]):
            sum_val_set[i] += val_df[i]
            count_val_set[i] += 1
    
    # Compute the average, handling cases where the count is zero
    avg_train_set = np.divide(sum_train_set, count_train_set, out=np.zeros_like(sum_train_set), where=count_train_set!=0)
    avg_val_set = np.divide(sum_val_set, count_val_set, out=np.zeros_like(sum_val_set), where=count_val_set!=0)
    
    # Turn the average train and validation sets into DataFrames
    avg_train_set = pd.DataFrame(avg_train_set, columns=headers)
    avg_val_set = pd.DataFrame(avg_val_set, columns=headers)

    # print header of the average train set and average validation DataFrame
    print("Average Train Set:") 
    print(avg_train_set.head())
    print("Average Validation Set:")
    print(avg_val_set.head())

    # Check the values of row 1000 of the average train set and average validation DataFrame
    print("Row 1000 of Average Train Set:")
    print(avg_train_set.iloc[1000])
    print("Row 1000 of Average Validation Set:")
    print(avg_val_set.iloc[1000]) 

    # separate the target variable from the features for each dataset
    # The target will contain Cycle and Normalized Discharge Capacity [-]
    # The features will contain Cycle with all other columns
    
    # Avg Train Set
    avg_train_set_y = avg_train_set[["Cycle", "Normalized Discharge Capacity [-]"]]
    avg_train_set_X = avg_train_set.drop(columns=["Normalized Discharge Capacity [-]"])

    # Avg Validation Set
    avg_val_set_y = avg_val_set[["Cycle", "Normalized Discharge Capacity [-]"]]
    avg_val_set_X = avg_val_set.drop(columns=["Normalized Discharge Capacity [-]"])

    return avg_train_set_X, avg_val_set_X, avg_train_set_y, avg_val_set_y



def create_models(train_set_X, val_set_X, train_set_y, val_set_y):
    """We will test linear regression, ridge regression, and lasso regression models.
    Ridge and Lasso will be tested for various alpha values [0.1, 0.25, 0.5, 0.75, 0.9]
    Create a dictionary to store the R^2 values"""
    r2_values = {}
    # Create a dictionary to store the models
    models = {}
    # Create a dictionary to store the model names
    model_names = {}
    
    for model_name in ["linear", "ridge", "lasso"]:
        r2_values[model_name] = []
        models[model_name] = []
        model_names[model_name] = []
        for alpha in [0.1, 0.25, 0.5, 0.75, 0.9]:
            if model_name == "linear":
                model = linear_model.LinearRegression()
            elif model_name == "ridge":
                model = linear_model.Ridge(alpha=alpha)
            elif model_name == "lasso":
                model = linear_model.Lasso(alpha=alpha)
            model.fit(train_set_X, train_set_y)
            r2 = model.score(val_set_X, val_set_y)
            r2_values[model_name].append(r2)
            models[model_name].append(model)
            if model_name == "linear":
                model_names[model_name].append(model_name)
            else:
                model_names[model_name].append(model_name + f"_{alpha}")

            # Print the R^2 values
            print(f"Model: {model_name}, alpha: {alpha}, R^2: {r2}")

    # print all dictionaries
    print("R^2 Values:")
    print(r2_values)
    print("Models:")
    print(models)
    print("Model Names:")
    print(model_names)
    
    # Find the best model
    best_r2 = -np.inf
    best_model = None
    best_model_name = None
    for model_name in r2_values.keys():
        for i, r2 in enumerate(r2_values[model_name]):
            if r2 > best_r2:
                best_r2 = r2
                best_model = models[model_name][i]
                best_model_name = model_names[model_name][i]
    print(f"Best Model: {best_model_name}")
    return best_model, best_model_name, best_r2


def test_model(best_model, best_model_name, data_real_driving):
    """Test the best model against all real driving data keys (8 in total)."""
    r2_values = {}
    # separate the target variable from the features for the real driving dataset
    # The target will contain Cycle and Normalized Discharge Capacity [-]
    # The features will contain Cycle with all other columns
    data_real_driving_y = {}
    data_real_driving_X = {}
    for key in data_real_driving.keys():
        data_real_driving_y[key] = data_real_driving[key][["Cycle", "Normalized Discharge Capacity [-]"]]
        data_real_driving_X[key] = data_real_driving[key].drop(columns=["Normalized Discharge Capacity [-]"])

    for key in data_real_driving.keys():
        r2 = best_model.score(data_real_driving_X[key], data_real_driving_y[key])
        r2_values[key] = r2
        print(f"Model: {best_model_name}, Key: {key}, R^2: {r2}")
    print("R^2 Values:")
    print(r2_values)
    avg_r2 = np.mean(list(r2_values.values()))
    print(f"Average R^2: {avg_r2}")
    return avg_r2


def main():
    # Load data
    data_constant_current, data_periodic, data_synthetic, data_real_driving = load_data()
    # Inspect dictionary keys
    inspect_all_data(data_constant_current, data_periodic, data_synthetic, data_real_driving)
    
    # Create train and validation sets for cc, p, and s data
    data_train_val = {'cc': data_constant_current, 'p': data_periodic, 's': data_synthetic}
    
    chosen_model= {}
    
    for key, data in data_train_val.items():
        # Create train and validation sets
        train_set_X, val_set_X, train_set_y, val_set_y = create_train_val_sets(data)
        # create linear, lasso, and ridge regression models - save R^2 values to a table
        best_model, best_model_name, best_r2_val = create_models(train_set_X, val_set_X, train_set_y, val_set_y)
        avg_r2_test = test_model(best_model, best_model_name, data_real_driving)
        chosen_model[key] = best_model, best_model_name, best_r2_val, avg_r2_test

    print(chosen_model)
    return


if __name__ == "__main__":
    main()