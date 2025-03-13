# ChemEng277 Regression Model Fitting for Various Battery Dynamic Cycling Procedures in Predicting Real-Driving Use Cases

Note: This GitHub Repo has been created for the sole purpose of the Stanford 2025 Winter Quarter ChemEng 277 course and will be used for academic purposes only.

PURPOSE:

    The purpose for this project is to test the predictiveness of several features measured either directly or indirectly in a seried of dynamic cycling experiments for State of Health (i.e. battery degradation). Currently, most industrial testing on battery health is done using constant current cycling. However, recent papers suggest that a dynamic cycling method (e.g. including rest intervals, etc) could reduce battery degradation. 


SOURCE DATA:

    All data was pulled from this Stanford Digital Repository link:
    https://purl.stanford.edu/td676xr4322

    Citation: Geslin, A., Xu, L., Ganapathi, D., Moy, K., Chueh, W., and Onori, S. (2024). Dataset - Dynamic cycling enhances battery lifetime. Stanford Digital Repository. Available at https://purl.stanford.edu/td676xr4322/version/1. https://doi.org/10.25740/td676xr4322.


RAW DATA DOWNLOADING & SAVING INSTRUCTIONS:

    - Please keep in mind that to do the above, you must individually download all files (187 in total, 184 of which are data - 92 raw data files and 92 aging data files) locally to your computer. These files are large in size (many > 100 MB) so, in order to upload to this GitHub Repository, we utilized Git Large File Storage (Git LFS). All files associated from the Geslin et al. source are located in the Data section of the Repo. Data/AgingData contains all 92 aging files. 

    - These files include Normalized Charge Capacity, Normalized Discharge Capacity, and Normalized Cumulative Capacity at the cycle level. Data/RawData contains all 92 raw cell data files. These files include a timer (in seconds), cycle, step, and loop identification as well as voltage, current, and capacity of the cell. Within Data/Other is the README from Geslin et al. as well as publishing_data_protocol_mapping_dic.json which has a dictionary of all the cell cycling protocols that were run and which cell number they correspond to. This file was directly consulted when determining how to bin each of the cell data in later files.


RAW DATA PROCESSING:

    - The RawData and AgingData of each cell are processed using the struct_data.py file. This file first imports the raw data and aging data and confirms that they can be merged. 

    - The raw data is compressed to the step level (lowest level) to gain extreme attributes and durations of individual actions. 

    - These features are then compared to the rest of the data from a given cycle and both averages, maximums, and durations are generated/compressed for the overal cycle level. The aging data is then merged with the processed raw data to create a new dataframe that contains all the processed data and aging data at the cycle level. 

    - The dataframes are then separated into the four recipe types: constant current, periodic, synthetic, and real driving. The dataframes are saved as dictionaries of dataframes (with keys identifying the recipe type) to pickle files for ease of use in subsequent python analysis files. These files are saved to Data/ProcessedData and will overwrite any same named files that were there previously. Because these files were used down the line and each test of this file generates new files, the four complete (all cycles accounted for) files were copied to Data/For_Analysis. 

    - As a recommendation with running the above file - please read the # NOTE on line 260. If you plan to generate the entire processed files again, keep nrows=NONE on line 261. This code will take several minutes to run through the large raw files. If you want to confirm structure or efficacy, I would recommend setting nrows=10000 and the code will complete within seconds.

    - Summary of the above: the file struct_data.py generates 4x files in the Data/ProcessedData folder. These files are copied over to the For_Analysis folder for further processing in the Data Analysis section below.

        - struct_data.py -> Data/ProcessedData


DATA ANALYSIS:

    - The overall goal of this analysis is to see how well three different battery cycling recipe types (constant current - CC, periodic current - P, and synthetic cycling - S) predict real driving battery SOH degradation. We identified the parameter "Normalized Discharge Capacity" to represent SOH, our label. We wanted to target SOH at particular cycle numbers, so our label became a 2-D array including both variables. 

    - There are several files that were created to analyze these data but they are all derived from the original model_analysis.py file. For this reason, the model_analysis.py file and its generated analysis section within the Data tab will be explained in full and all subsequent files will speak to the differences from the original file. 

    - The model_analysis.py file first loads in all of the Processed Data locatee in Data/For_Analysis then inspects the data through inspect_dictionary and inspect_all_data for debugging purposes. 

    - Initially, it was not incorporated but the next function run is remove_outliers because some Normalized Discharge Capacity (SOH) values are uncharacteristically high which showed to have a significant impact on model fitting. A median of closely surrounding data for all features and labels is taken if the SOH is greater than a given THRESHOLD in order to muffle extreme outliers. 

    - Once the data is processed, it is ready to be analyzed. We took the approach of training and validating models of each recipe type and then testing on each individual real driving data set. That is for example -  we take all cell cycling data for continuous current, split it into a 80% train, 20% validation set using the create_train_val_sets function and then take our best model and test its prediction accuracy with the 8 real driving cells. 
    
    - For each train/validation set, we performed Linear, Lasso, and Ridge Regression at various hyperparameter values. The RobustScaler function from sklearn was utilized to also muffle the impact of outliers in the model. Initially, we used StandardScaler but following our findings of impactful outliers, we modified our code to accommodate this function instead. 
    
    - For each recipe type (CC, P, or S), we identified the best performing model by calculating R^2 of the validation set when tested on the training model using the create_models function. One of the dataframes generated from this function is the coefficients_df. These were saved for each recipe type and uploaded to Data/Completed_Analysis as .csv files. These files contain model names, hyperparameters, R^2 of validation test sets, and coefficients for each feature when predicting for SOH. This file is very helpful in identifying whether a feature is overpowering other features. Additionally, within the create_models function, we check the features in the train sets for multicollinearity via the check_multicollinearity function. This function is saved as a heatmap png figure to Data/Completed_Analysis. 
    
    - The identified "best models" for each recipe type were fed into the test_model function which tests the best regression models on the 8 real driving cycling data sets. R^2 for each of these sets are calculated as well as the average R^2 of the model. Once all three recipe types have completed the loop, a chosen_model dataframe is created and saved to a csv file within Data/Completed_Analysis. This file identifies which model in each recipe type was chosen as the best model, how the model fared in validation (R^2), how the model fared on average in the real driving test sets, and the R^2 for each individual test data set (8 total). 
    
    - Additionally, within the test_models function, a function named plot_figures is called which will show the predicted and true values for Normalized Discharge Capacity at each Cycle number. These figures can be used to visually understand the R^2 values calculated for the test cases as well as any anomalies with predictions. To make it clear - cells numbered 89-96 were the ones that followed the real driving protocols.

    - These initial models ran into several roadblocks - namely with multicollinearity as can be found in the Correlation_Matrix.png. Multicollinearity is best adjusted with Ridge Regression when compared to Lasso and Linear Regression. As such, a new file named model_analysis_ridge_only.py was created. This file has an identical structure to model_analysis.py and calls the same files that were prepared in struct_data.py. A new set of alpha values across a larger value range were tested in this model to identify the best model. In this case, we still had a 2-D label array including both Cycle and Normalized Discharge Capacity. All generated files and figures were saved to the Data/Completed_Analysis_Ridge_Only folder.

    - Multicollinearity continued to persist in this model so a subsequent file - model_analysis_huber_only.py was created to test the three recipe types with Huber regression across various epsilon hyperparameters. Huber Regression is particularly good at avoiding overfitting to outliers in process data. The caveat of Huber Regression is that the model expects a 1-D label. Since our overall goal was to predict SOH, we changed the y labels to only include Normalized Discharge Capacity. All other parameters were captured as featured just like in previous files. All generated files were saved to Data/Completed_Analysis_Huber_Only.

    - Results from this method were suboptimal with predictions for all but validation of constant current. However, when inspecting the coefficients for constant current in the cc_coefficients.csv file, we noticed that the predicitons for SOH were not overpowered by Cycle. The 1-D label method was used to update the Ridge Regression only code in the file model_analysis_ridge_one_y.py. All generated files were saved to Data/Completed_Analysis_Ridge_one_y. The positives from this data were enhanced prediction of validation sets, however the test sets continued to have suboptimal predictions on average. The correlation matrix generated from these data also showed to have lowered multicollinearity when testing for only SOH as the label.

    - Summary of above: All above python .py files are saved to the same directory level as this readme. All data generated is saved in associated folders to the Data folder. The below is organized in the python file -> Data Folder format

        - model_analysis.py -> Data/Completed_Analysis
        - model_analysis_ridge_only.py -> Data/Completed_Analysis_Ridge_Only
        - model_analysis_huber_only.py -> Data/Completed_Analysis_Huber_Only
        - model_analysis_ridge_one_y.py -> Data/Completed_Analysis_Ridge_one_y

    Data folders include the following files:

        - For each recipe (cc, p, s) type: "recipe"_coefficients.csv, "recipe"_Model_Predictions_vs_True_Values_Cell_##
        - One cohesive file: chosen_model.csv


UPDATED ANALYSIS

    - Upon analyzing results from the above files, revisions to the two original files were made to reduce multicollinearity. The structure of these updated files are identical to struct_data.py and model_analysis.py with a few exceptions. 
    
    - The file struct_data_truncated.py reads and processes the raw data identically, however, it omits the inclusion of time-based features: Total Time, Rest Time, Charge Time, and Discharge Time. As these features could be consolidated into Rest Fraction, Charge Fraction, and Discharge Fraction, they were actually impacting model performance. Ridge Regression is meant to account for correlated features but the team wanted to confirm whether omission of these features would improve performance. The four pickle files generated from this file were saved to Data/ProcessedData_Truncated. 
    
    - The subsequent file, model_analysis_truncated.py calls the files generated in Data/ProcessedData_Truncated and processes the files using the same structure as model_analysis.py by train/validating SOH R^2 values for the three recipe types using Linear, Lasso, and Ridge Regression across several hyperparameters. The the model chooses the best R^2 value for each recipe and proceeds to test predictivity for the 8 real driving cycling data cases. This file has fewer features than model_analysis.py because it calls generated files that have less features to begin with. Additionally, this file focuses on testing for a 1-D label (SOH) rather than the 2-D label (Cycle & SOH) implemented for model_analysis.py. The generated files follow the same structure as explained in the Analysis section and are saved to Data/Completed_Analysis_Truncated.

    - Summary of above:

        - Cleaning Data: struct_data_truncated.py -> Data/ProcessedData_Truncated
        - Analysis of Data: model_analysis_truncated.py -> Data/Completed_Analysis_Truncated