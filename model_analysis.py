import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import RobustScaler
import seaborn as sns

"""As a note for data analysis: all models will be trained and validated individually for each of the three 
datasets: constant current, periodic, and synthetic. The real driving dataset will be used for testing purposes
only. Normalized Discharge Capacity (or State of Health - "SOH") will be the target (y) variable for all models.
Once random keys are determined for each dataset, those keys will be averaged to create one 2D list of parameters
versus cycle number. The same method will be used for the validation set. The target variable matrix will be
2D with SOH vs cycle number. These models will be fitted to Linear, Lasso, and Ridge Regression. Lasso Regression 
will be tested for various alpha (lambda) hyperparameters [0.1, 0.25, 0.5, 0.75, 0.9]. Ridge Regression will be
tested for various alpha hyperparameters [10, 25, 50, 75, 90]. The Linear Regression model will be tested as is.
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

def remove_outliers(data):
    """Replace outliers from the data with data that fits the discharge profile. Some of the normalized 
    discharge capacities are greater than 2.0. These data should be replaced with a value that is the working 
    median of the 10 cycles before and after the cycle in question. In cases where the cycle is within the 
    first 10 cycles, the median of the 10 cycles after the cycle in question should be used. In cases where 
    the cycle is within the last 10 cycles, the median of the 10 cycles before the cycle in question should 
    be used. All other columns will perform the same computation at the identified cycle. In all cases, cycle 
    number should remain the same."""

    new_data = {}
    THRESHOLD = 2.0
    for key in data.keys():
        # The target will contain Cycle and Normalized Discharge Capacity [-]
        # The features will contain Cycle with all other columns
        data_cycle = data[key]["Cycle"]
        data_other = data[key].drop(columns=["Cycle"])
        for i in range(10, len(data_other) - 10):
            if data_other.loc[i, "Normalized Discharge Capacity [-]"] > THRESHOLD:
                # perform the same computation at the identified cycle for all columns
                for col in data_other.columns:
                    data_other.loc[i, col] = np.median(data_other.loc[i-10:i+10, col])
        for i in range(10):
            if data_other.loc[i, "Normalized Discharge Capacity [-]"] > THRESHOLD:
                for col in data_other.columns:
                    data_other.loc[i, col] = np.median(data_other.loc[i:i+10, col])
        for i in range(len(data_other) - 10, len(data_other)):
            if data_other.loc[i, "Normalized Discharge Capacity [-]"] > THRESHOLD:
                for col in data_other.columns:
                    data_other.loc[i, col] = np.median(data_other.loc[i-10:i, col])
        # add the cycle data back to the DataFrame
        new_data[key] = pd.concat([data_cycle, data_other], axis=1)
        
    return new_data


def create_train_val_sets(data, seed=77):
    # Save the headers of the DataFrames
    headers = data[list(data.keys())[0]].columns

    np.random.seed(seed)

    # randomly shuffle the data, split into train and validation sets, and average the train and validation sets
    # add all the data.keys() into a numpy array and shuffle the array
    data_keys = np.array(list(data.keys()))
    np.random.shuffle(data_keys)

    # add the first 80% of the shuffled array to the train set and the last 20% to the validation set
    train_keys = data_keys[:int(0.8 * len(data_keys))]
    val_keys = data_keys[int(0.8 * len(data_keys)):]

    # Create a dictionary to store the train and validation sets
    train_set = {}
    val_set = {}
    # Add the data to the train and validation sets
    for key in train_keys:
        train_set[key] = data[key]
    for key in val_keys:
        val_set[key] = data[key]


    # Determine the maximum number of rows across all DataFrames
    max_rows = max(df.shape[0] for df in data.values())
    
    # Initialize sum and count arrays
    sum_train_set = np.zeros((max_rows, data[list(data.keys())[0]].shape[1]))
    count_train_set = np.zeros((max_rows, 1))
    sum_val_set = np.zeros((max_rows, data[list(data.keys())[0]].shape[1]))
    count_val_set = np.zeros((max_rows, 1))
    
    # Sum the train and validation sets and count the occurrences
    for key in train_set.keys():
        train_df = train_set[key].values
        for i in range(train_df.shape[0]):
            sum_train_set[i] += train_df[i]
            count_train_set[i] += 1

    for key in val_set.keys():
        val_df = val_set[key].values
        for i in range(val_df.shape[0]):
            sum_val_set[i] += val_df[i]
            count_val_set[i] += 1

    # Create a dictionary to store the average train and validation sets
    avg_train_set = {}
    avg_val_set = {}
    
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
    if len(avg_train_set) > 999:
        print("Row 1000 of Average Train Set:")
        print(avg_train_set.iloc[999])
    if len(avg_val_set) > 999:
        print("Row 1000 of Average Validation Set:")
        print(avg_val_set.iloc[999]) 

    # separate the target variable from the features for each dataset
    # The target will contain Cycle and Normalized Discharge Capacity [-]
    # The features will contain Cycle with all other columns
    
    # Avg Train Set
    avg_train_set_y = avg_train_set[["Cycle", "Normalized Discharge Capacity [-]"]]
    avg_train_set_X = avg_train_set.drop(columns=["Normalized Discharge Capacity [-]"])

    # print headers to avg_train set_y
    print("Average Train Set y: ")
    print(avg_train_set_y.head())


    # Avg Validation Set
    avg_val_set_y = avg_val_set[["Cycle", "Normalized Discharge Capacity [-]"]]
    avg_val_set_X = avg_val_set.drop(columns=["Normalized Discharge Capacity [-]"])

    # Update headers
    headers_x = avg_train_set_X.columns
    headers_y = avg_train_set_y.columns

    return avg_train_set_X, avg_val_set_X, avg_train_set_y, avg_val_set_y, headers_x

def remove_before_char(s, char):
    # Split the string at the specified character
    parts = s.split(char, 1)
    # Return the part after the character, or the original string if the character is not found
    return parts[1] if len(parts) > 1 else s

def create_models(train_set_X, val_set_X, train_set_y, val_set_y, headers_x, key):
    """We will test linear regression, ridge regression, and lasso regression models.
    Lasso will be tested for various alpha values [0.1, 0.25, 0.5, 0.75, 0.9]
    Ridge will be tested for various alpha values [10, 25, 50, 75, 90]
    Create a dictionary to store the R^2 values"""
    
    print(f"Creating models for {key} data.")
    
    # Scale the X data
    scaler = RobustScaler()
    train_set_X = scaler.fit_transform(train_set_X)
    val_set_X = scaler.transform(val_set_X)

    # print the first few rows of the scaled train and validation sets X
    print("Scaled Train Set X:")
    print(train_set_X[:5])
    print("Scaled Validation Set X:")
    print(val_set_X[:5])

    # Check for multicollinearity
    check_multicollinearity(pd.DataFrame(train_set_X, columns=headers_x))

    # Scale the y data
    target_scaler = RobustScaler()
    train_set_y = target_scaler.fit_transform(train_set_y)
    val_set_y = target_scaler.transform(val_set_y)

    # print the first few rows of the scaled train and validation sets Y
    print("Scaled Train Set y:")
    print(train_set_y[:5])
    print("Scaled Validation Set y:")
    print(val_set_y[:5])

    # Print shapes of the datasets
    print(f"Shape of train_set_X: {train_set_X.shape}")
    print(f"Shape of val_set_X: {val_set_X.shape}")
    print(f"Shape of train_set_y: {train_set_y.shape}")
    print(f"Shape of val_set_y: {val_set_y.shape}")

    r2_values = {}
    # Create a dictionary to store the models
    models = {}
    # Create a dictionary to store the model names
    model_names = {}
    coefficients_list = []  # List to store coefficients

    MULTIPLIER = 100
    
    for model_name in ["linear", "ridge", "lasso"]:
        r2_values[model_name] = []
        models[model_name] = []
        model_names[model_name] = []
        for alpha in [0.1, 0.25, 0.5, 0.75, 0.9]:
            if model_name == "linear":
                model = linear_model.LinearRegression()
                alpha_value = None
            elif model_name == "ridge":
                model = linear_model.Ridge(alpha=MULTIPLIER*alpha, max_iter=10000)
                alpha_value = MULTIPLIER*alpha
            elif model_name == "lasso":
                model = linear_model.Lasso(alpha=alpha, max_iter=10000)
                alpha_value = alpha
            model.fit(train_set_X, train_set_y)
            r2 = model.score(val_set_X, val_set_y)
            r2_values[model_name].append(r2)
            models[model_name].append(model)
            if model_name == "linear":
                model_names[model_name].append(model_name)
                print(f"Model: {model_name}, R^2: {r2}")
            elif model_name == "ridge":
                model_names[model_name].append(model_name + f"_{MULTIPLIER*alpha}")
                print(f"Model: {model_name}, alpha: {MULTIPLIER*alpha}, R^2: {r2}")
            else:
                model_names[model_name].append(model_name + f"_{alpha}")
                print(f"Model: {model_name}, alpha: {alpha}, R^2: {r2}")

            # Store coefficients
            coef_dict = {
                "model_name": model_name,
                "alpha": alpha_value,
                "R^2": r2
            }
            # Use zip to pair feature names with their coefficients
            coef_dict.update({f: coef for f, coef in zip(headers_x, model.coef_.flatten())})
            print(f'Size of Coefficients structure = {len(model.coef_)}')
            coefficients_list.append(coef_dict)

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
    best_alpha = None
    for model_name in r2_values.keys():
        for i, r2 in enumerate(r2_values[model_name]):
            if r2 > best_r2:
                best_r2 = r2
                best_model = models[model_name][i]
                best_model_name = model_names[model_name][i]
                best_alpha = remove_before_char(str(model_names[model_name][i]), "_")
                # 10 * alpha if model_name == "ridge" else alpha
    print(f"Best Model: {best_model_name}")

    # Create a DataFrame for the coefficients
    coefficients_df = pd.DataFrame(coefficients_list)
    print("Coefficients DataFrame:")
    print(coefficients_df)

    return best_model, best_model_name, best_alpha, best_r2, scaler, target_scaler, coefficients_df

def plot_figures(y_true, y_pred, key, recipe_type, target_scaler):
    """Plot the predictions against the actual values."""
    # Inverse transform the target variable
    y_true = target_scaler.inverse_transform(y_true)
    y_pred = target_scaler.inverse_transform(y_pred)
    
    # add headers to the y_pred DataFrame where the first column is "Cycle" and the second column is "Normalized Discharge Capacity [-]"
    y_pred = pd.DataFrame(y_pred[:,1], columns=["Normalized Discharge Capacity [-]"])
    y_true = pd.DataFrame(y_true, columns=["Cycle", "Normalized Discharge Capacity [-]"])

    plt.figure(figsize=(10, 6))
    # scatter plot of the true values
    plt.scatter(y_true["Cycle"], y_true["Normalized Discharge Capacity [-]"], label="True", color="orange")
    # line plot of the predicted values    
    plt.plot(y_true["Cycle"], y_pred, label="Predicted", color="blue")
    plt.xlabel("Cycle")
    plt.ylabel("Normalized Discharge Capacity [-]")
    plt.title(f"{recipe_type} Model Predictions vs. True Values for Cell #{int(key) + 89}")
    plt.legend(["True", "Predicted"])
    # save the figure to Data/Completed_Analysis
    plt.savefig(f"Data/Completed_Analysis/{recipe_type}_Model_Predictions_vs_True_Values_Cell_{int(key) + 89}.png")
    plt.close()  # Close the figure to free up memory
    return

def test_model(best_model, best_model_name, data_real_driving, scaler, target_scaler, recipe_type):
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
        # Apply the same scaling to the test data
        data_real_driving_X[key] = scaler.transform(data_real_driving_X[key])
        data_real_driving_y[key] = target_scaler.transform(data_real_driving_y[key])
        # Calculate R^2
        r2 = best_model.score(data_real_driving_X[key], data_real_driving_y[key])
        r2_values[key] = r2
        # print the R^2 value for each key
        print(f"Model: {best_model_name}, Cell #: {int(key) + 89}, R^2: {r2}")
        # make predictions
        y_pred = best_model.predict(data_real_driving_X[key])
        # print the first few predictions
        print("Predictions:")
        print(y_pred[:5])
        # print the shape of the predictions
        print(f"Shape of Predictions: {y_pred.shape}")
        # plot the predictions against the actual values
        plot_figures(data_real_driving_y[key], y_pred, key, recipe_type, target_scaler)
    # print all R^2 values
    print("R^2 Values:")
    print(r2_values)
    avg_r2 = np.mean(list(r2_values.values()))
    print(f"Average R^2: {avg_r2}")
    return avg_r2, r2_values

def check_multicollinearity(data):
    # Calculate the correlation matrix
    corr_matrix = data.corr()
    # Plot the heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    # Adjust layout to ensure nothing is cut off
    plt.tight_layout()
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.3)
    # Save the figure to Data/Completed_Analysis
    plt.savefig("Data/Completed_Analysis/Correlation_Matrix.png")
    plt.close()  # Close the figure to free up memory and avoid overlay plots
    return

def main():
    # Load data
    data_constant_current, data_periodic, data_synthetic, data_real_driving = load_data()
    # Inspect dictionary keys
    inspect_all_data(data_constant_current, data_periodic, data_synthetic, data_real_driving)

    # remove discharge data outliers from all datasets
    data_constant_current = remove_outliers(data_constant_current)
    data_periodic = remove_outliers(data_periodic)
    data_synthetic = remove_outliers(data_synthetic)
    data_real_driving = remove_outliers(data_real_driving)
    
    # Create train and validation sets for cc, p, and s data
    data_train_val = {'cc': data_constant_current, 'p': data_periodic, 's': data_synthetic}
    
    chosen_model= {}
    
    for key, data in data_train_val.items():
        # Create train and validation sets
        train_set_X, val_set_X, train_set_y, val_set_y, headers_x = create_train_val_sets(data)
        # create linear, lasso, and ridge regression models - save R^2 values to a table
        best_model, best_model_name, best_alpha, best_r2_val, scaler, target_scaler, coefficients_df = create_models(train_set_X, val_set_X, train_set_y, val_set_y, headers_x, key)
        # save the coefficients_df to a csv file to Data/Completed_Analysis
        coefficients_df.to_csv(f"Data/Completed_Analysis/{key}_coefficients.csv", index=False)
        # test the best model against the real driving data
        avg_r2_test, r2_values = test_model(best_model, best_model_name, data_real_driving, scaler, target_scaler, key)
        chosen_model[key] = {
            "Model": best_model_name,
            "Alpha": best_alpha,
            "R^2 Validation": best_r2_val,
            "Avg R^2 Real Driving Test Set": avg_r2_test
        }
        # add the r2 values from the r2_values dictionary to the chosen_model dictionary
        chosen_model[key].update(r2_values)  # Corrected line
        

    # turn the chosen_model dictionary into a DataFrame
    chosen_model_df = pd.DataFrame.from_dict(chosen_model, orient='index').reset_index()
    chosen_model_df.columns = ["Recipe Type", "Model", "Alpha", "R^2 Validation", "Avg R^2 Real Driving Test Set"] + [89 + int(i) for i in list(r2_values)]

    # save the chosen_model DataFrame to a csv file to Data/Completed_Analysis
    chosen_model_df.to_csv("Data/Completed_Analysis/chosen_model.csv", index=False)
    print(chosen_model_df)

    """As a note: Upon inspection of the above data, ridge regression with an alpha value of 90 from the 
    periodic data set should also be tested on the real driving data set. As a result, the above code will be 
    repeated in a subsequent file."""

    return

if __name__ == "__main__":
    main()