import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import seaborn as sns
from sklearn.model_selection import KFold


# Load data (.pkl dictionary files) from Data/For_Analysis
def load_data():
    with open("Data/ProcessedData_Truncated/df_constant_current.pkl", "rb") as f:
        data_constant_current = pickle.load(f)
    with open("Data/ProcessedData_Truncated/df_periodic.pkl", "rb") as f:
        data_periodic = pickle.load(f)
    with open("Data/ProcessedData_Truncated/df_synthetic.pkl", "rb") as f:
        data_synthetic = pickle.load(f)
    with open("Data/ProcessedData_Truncated/df_real_driving.pkl", "rb") as f:
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
    # print("Average Train Set:") 
    # print(avg_train_set.head())
    # print("Average Validation Set:")
    # print(avg_val_set.head())

    # Check the values of row 1000 of the average train set and average validation DataFrame
    # if len(avg_train_set) > 999:
    #     print("Row 1000 of Average Train Set:")
    #     print(avg_train_set.iloc[999])
    # if len(avg_val_set) > 999:
    #     print("Row 1000 of Average Validation Set:")
    #     print(avg_val_set.iloc[999]) 

    # separate the target variable from the features for each dataset
    # The target will contain Cycle and Normalized Discharge Capacity [-]
    # The features will contain Cycle with all other columns
    
    # Avg Train Set
    avg_train_set_y = avg_train_set["Normalized Discharge Capacity [-]"]
    avg_train_set_X = avg_train_set.drop(columns=["Normalized Discharge Capacity [-]"])

    # print headers to avg_train set_y
    # print("Average Train Set y: ")
    # print(avg_train_set_y.head())


    # Avg Validation Set
    avg_val_set_y = avg_val_set["Normalized Discharge Capacity [-]"]
    avg_val_set_X = avg_val_set.drop(columns=["Normalized Discharge Capacity [-]"])

    # Update headers
    headers_x = avg_train_set_X.columns
    # headers_y = avg_train_set_y.columns

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
    
    # print(f"Creating models for {key} data.")
    
    # Scale the X data
    scaler = RobustScaler()
    train_set_X = scaler.fit_transform(train_set_X)
    val_set_X = scaler.transform(val_set_X)

    # print the first few rows of the scaled train and validation sets X
    # print("Scaled Train Set X:")
    # print(train_set_X[:5])
    # print("Scaled Validation Set X:")
    # print(val_set_X[:5])

    # Check for multicollinearity
    check_multicollinearity(pd.DataFrame(train_set_X, columns=headers_x))

    # Scale the y data
    target_scaler = RobustScaler()
    train_set_y = target_scaler.fit_transform(train_set_y.values.reshape(-1, 1)).flatten()
    val_set_y = target_scaler.transform(val_set_y.values.reshape(-1, 1)).flatten()

    # print the first few rows of the scaled train and validation sets Y
    # print("Scaled Train Set y:")
    # print(train_set_y[:5])
    # print("Scaled Validation Set y:")
    # print(val_set_y[:5])

    # Print shapes of the datasets
    # print(f"Shape of train_set_X: {train_set_X.shape}")
    # print(f"Shape of val_set_X: {val_set_X.shape}")
    # print(f"Shape of train_set_y: {train_set_y.shape}")
    # print(f"Shape of val_set_y: {val_set_y.shape}")

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
                #print(f"Model: {model_name}, R^2: {r2}")
            elif model_name == "ridge":
                model_names[model_name].append(model_name + f"_{MULTIPLIER*alpha}")
                #print(f"Model: {model_name}, alpha: {MULTIPLIER*alpha}, R^2: {r2}")
            else:
                model_names[model_name].append(model_name + f"_{alpha}")
                #print(f"Model: {model_name}, alpha: {alpha}, R^2: {r2}")

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
                
    print(f"Best Model: {best_model_name}")

    # Create a DataFrame for the coefficients
    coefficients_df = pd.DataFrame(coefficients_list)
    print("Coefficients DataFrame:")
    print(coefficients_df)

    return best_model, best_model_name, best_alpha, best_r2, scaler, target_scaler, coefficients_df

def create_models_cv(train_set_X, val_set_X, train_set_y, val_set_y, headers_x, seed=77):
    """Modified version of create_models, this function performs 5-fold cross validation on the training set.
    It also trains linear, ridge, and lasso models (with the same alpha values) on the training folds,
    calcs the average CV R^2, selects the best model, retrains it on the entire training set,
    and evaluates on the hold-out validation set."""

    # Initialize KFold cross-validation with 5 splits
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    # Scale the X data
    scaler = RobustScaler()
    train_set_X_scaled = scaler.fit_transform(train_set_X)
    val_set_X_scaled = scaler.transform(val_set_X)

    # Scale the y data
    target_scaler = RobustScaler()
    train_set_y_scaled = target_scaler.fit_transform(train_set_y.values.reshape(-1, 1)).flatten()
    val_set_y_scaled = target_scaler.transform(val_set_y.values.reshape(-1, 1)).flatten()

    # Dictionaries to store results for each model type and parameter value
    r2_values = {}
    models = {}
    model_names = {}
    coefficients_list = []
    MULTIPLIER = 100  # Used for Ridge regression alpha scaling

    for model_type in ["linear", "ridge", "lasso"]:
        r2_values[model_type] = []
        models[model_type] = []
        model_names[model_type] = []
        for alpha in [0.1, 0.25, 0.5, 0.75, 0.9]:
            cv_scores = []
            # For each fold in the CV process:
            for train_index, cv_val_index in kf.split(train_set_X_scaled):
                X_cv_train = train_set_X_scaled[train_index]
                X_cv_val = train_set_X_scaled[cv_val_index]
                y_cv_train = train_set_y_scaled[train_index]
                y_cv_val = train_set_y_scaled[cv_val_index]

                # Initialize model based on type and alpha
                if model_type == "linear":
                    model = linear_model.LinearRegression()
                    alpha_value = None
                elif model_type == "ridge":
                    # Scale alpha for ridge if needed
                    model = linear_model.Ridge(alpha=MULTIPLIER * alpha, max_iter=10000)
                    alpha_value = MULTIPLIER * alpha
                elif model_type == "lasso":
                    model = linear_model.Lasso(alpha=alpha, max_iter=10000)
                    alpha_value = alpha

                # Fit model on the current training fold and score on the CV validation fold
                model.fit(X_cv_train, y_cv_train)
                cv_score = model.score(X_cv_val, y_cv_val)
                cv_scores.append(cv_score)

            # Average CV R^2 score across the 5 folds for this model candidate
            avg_cv_score = np.mean(cv_scores)
            r2_values[model_type].append(avg_cv_score)
            models[model_type].append(model)  # Note: model from the last fold
            if model_type == "linear":
                model_names[model_type].append(model_type)
            elif model_type == "ridge":
                model_names[model_type].append(f"{model_type}_{MULTIPLIER * alpha}")
            else:
                model_names[model_type].append(f"{model_type}_{alpha}")

            # Store coefficients (from the last fold model fit)
            coef_dict = {
            
                "model_type": model_type,
                "alpha": alpha_value,
                "CV_R^2": avg_cv_score
            }
            # Pair each feature name with its corresponding coefficient value
            coef_dict.update({feature: coef for feature, coef in zip(headers_x, model.coef_.flatten())})
            coefficients_list.append(coef_dict)

    # Determine the best model based on the average cross validation score
    best_cv_r2 = 0
    best_model = None
    best_model_name = None
    best_alpha = None

    for model_type in r2_values.keys():
        for i, cv_score in enumerate(r2_values[model_type]):
            if cv_score > best_cv_r2:
                best_cv_r2 = cv_score
                best_model = models[model_type][i]
                best_model_name = model_names[model_type][i]
                best_alpha = None if model_type == "linear" else remove_before_char(str(model_names[model_type][i]), "_")

    print(f"Best CV Model: {best_model_name} with average CV R^2: {best_cv_r2}")

    # Retrain the best model on the entire training set using the same scaling parameters
    if "linear" in best_model_name:
        best_model_retrained = linear_model.LinearRegression()
    elif "ridge" in best_model_name:
        best_model_retrained = linear_model.Ridge(alpha=float(best_alpha), max_iter=10000)
    elif "lasso" in best_model_name:
        best_model_retrained = linear_model.Lasso(alpha=float(best_alpha), max_iter=10000)

    best_model_retrained.fit(train_set_X_scaled, train_set_y_scaled)
    final_score = best_model_retrained.score(val_set_X_scaled, val_set_y_scaled)
    #print(f"Final Validation R^2 on hold-out set: {final_score}")

    # Create a DataFrame to display all coefficients from each candidate
    coefficients_df = pd.DataFrame(coefficients_list)

    return best_model_retrained, best_model_name, best_alpha, best_cv_r2, final_score, scaler, target_scaler, coefficients_df

def plot_figures(y_true, y_pred, key, recipe_type, target_scaler, x_true, scaler):
    """Plot the predictions against the actual values."""
    # Inverse transform the features
    x_true = scaler.inverse_transform(x_true)

    # Inverse transform the target variable
    y_true = target_scaler.inverse_transform(y_true.reshape(-1, 1))
    y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1))
    
    # add headers to the y_pred DataFrame where the first column is "Cycle" and the second column is "Normalized Discharge Capacity [-]"
    y_pred = pd.DataFrame(y_pred, columns=["Normalized Discharge Capacity [-]"])
    y_true = pd.DataFrame(y_true, columns=["Normalized Discharge Capacity [-]"])
    

    plt.figure(figsize=(10, 6))
    # scatter plot of the true values
    plt.scatter(x_true[:, 0], y_true["Normalized Discharge Capacity [-]"], label="True", color="orange")
    # line plot of the predicted values    
    plt.plot(x_true[:, 0], y_pred, label="Predicted", color="blue")
    plt.xlabel("Cycle")
    plt.ylabel("Normalized Discharge Capacity [-]")
    plt.title(f"{recipe_type} Model Predictions vs. True Values for Cell #{int(key) + 89}")
    plt.legend(["True", "Predicted"])
    # save the figure to Data/Completed_Analysis
    plt.savefig(f"Data/Completed_Analysis_Truncated/{recipe_type}_Model_Predictions_vs_True_Values_Cell_{int(key) + 89}.png")
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
        data_real_driving_y[key] = data_real_driving[key]["Normalized Discharge Capacity [-]"]
        data_real_driving_X[key] = data_real_driving[key].drop(columns=["Normalized Discharge Capacity [-]"])

    for key in data_real_driving.keys():
        # Apply the same scaling to the test data
        data_real_driving_X[key] = scaler.transform(data_real_driving_X[key])
        data_real_driving_y[key] = target_scaler.transform(data_real_driving_y[key].values.reshape(-1, 1)).flatten()
        # Calculate R^2
        r2 = best_model.score(data_real_driving_X[key], data_real_driving_y[key])
        r2_values[key] = r2
        # print the R^2 value for each key
        #print(f"Model: {best_model_name}, Cell #: {int(key) + 89}, R^2: {r2}")
        # make predictions
        y_pred = best_model.predict(data_real_driving_X[key])
        # print the first few predictions
        #print("Predictions:")
        #print(y_pred[:5])
        # print the shape of the predictions
        #print(f"Shape of Predictions: {y_pred.shape}")
        # plot the predictions against the actual values
        plot_figures(data_real_driving_y[key], y_pred, key, recipe_type, target_scaler, data_real_driving_X[key], scaler)
    # print all R^2 values
    #print("R^2 Values:")
    #print(r2_values)
    avg_r2 = np.mean(list(r2_values.values()))
    #print(f"Average R^2: {avg_r2}")

    #Concat dictionaries to one df for X and Y
    test_X_df = np.vstack(list(data_real_driving_X[key]))
    test_y_df = np.vstack(list(data_real_driving_y[key]))
    return avg_r2, test_X_df, test_y_df

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
    # Save the figure to Data/Completed_Analysis_Truncated
    plt.savefig("Data/Completed_Analysis_Truncated/Correlation_Matrix.png")
    plt.close()  # Close the figure to free up memory and avoid overlay plots
    return

def main():
    # Load data
    data_constant_current, data_periodic, data_synthetic, data_real_driving = load_data()
    # Inspect dictionary keys
    inspect_all_data(data_constant_current, data_periodic, data_synthetic, data_real_driving)
    # Create train and validation sets for cc, p, and s data
    data_train_val = {'cc': data_constant_current, 'p': data_periodic, 's': data_synthetic}
    
    chosen_model= {}
    chosen_model_cv= {}
    best_models = {}

    for key, data in data_train_val.items():
        print("type of data evaluated",key)
        # Create train and validation sets
        train_set_X, val_set_X, train_set_y, val_set_y, headers_x = create_train_val_sets(data)
        # create linear, lasso, and ridge regression models - save R^2 values to a table
        best_model, best_model_name, best_alpha, best_r2_val, scaler, target_scaler, coefficients_df = create_models(train_set_X, val_set_X, train_set_y, val_set_y, headers_x, key)
        best_model_cv, best_model_name_cv, best_alpha_cv, best_r2_cv, final_score_cv, scaler_cv, target_scaler_cv, coefficients_df_cv = create_models_cv(train_set_X, val_set_X, train_set_y, val_set_y, headers_x, seed=77)
        # save the coefficients_df to a csv file to Data/Completed_Analysis_Truncated
        coefficients_df.to_csv(f"Data/Completed_Analysis_Truncated/{key}_coefficients.csv", index=False)
        coefficients_df_cv.to_csv(f"Data/Completed_Analysis_Truncated/{key}_coefficients.csv", index=False)
        # test the best model against the real driving data
        avg_r2_test, test_X_df, test_y_df = test_model(best_model, best_model_name, data_real_driving, scaler, target_scaler, key)
        chosen_model[key] = {
            "Model": best_model_name,
            "Alpha": best_alpha,
            "R^2 Validation": best_r2_val,
            "Avg R^2 Real Driving Test Set": avg_r2_test
        }
        avg_r2_test_cv, test_X_df_cv, test_y_df_cv = test_model(best_model_cv, best_model_name_cv, data_real_driving, scaler_cv, target_scaler_cv, key)
        chosen_model_cv[key] = {
            "Model_CV": best_model_name_cv,
            "Alpha_CV": best_alpha_cv,
            "R^2 Validation_CV": best_r2_cv,
            "Avg R^2 Real Driving Test Set_CV": avg_r2_test_cv
        }
        #Create df with best models for each Cycling type: cc, p and s; where x is the best model data and y the y predict
        # turn the chosen_model dictionary into a DataFrame
        chosen_model_df = pd.DataFrame.from_dict(chosen_model, orient='index').reset_index()
        chosen_model_df.columns = ["Recipe Type", "Model", "Alpha", "R^2 Validation", "Avg R^2 Real Driving Test Set"]
        chosen_model_df_cv = pd.DataFrame.from_dict(chosen_model, orient='index').reset_index()
        chosen_model_df_cv.columns = ["Recipe Type", "Model", "Alpha", "R^2 Validation", "Avg R^2 Real Driving Test Set"]

        # save the chosen_model DataFrame to a csv file to Data/Completed_Analysis_Truncated
        chosen_model_df.to_csv("Data/Completed_Analysis_Truncated/chosen_model.csv", index=False)
        chosen_model_df_cv.to_csv("Data/Completed_Analysis_Truncated/chosen_model.csv", index=False)
        print("chosen_model_df",chosen_model_df)
        print("chosen_model_df_cv", chosen_model_df_cv)
        return


if __name__ == "__main__":
    main()