# ChemEng277 Regression Model Fitting for Various Battery Dynamic Cycling Procedures in Predicting Real-Driving Use Cases

Note: This GitHub Repo has been created for the sole purpose of the Stanford 2025 Winter Quarter ChemEng 277 course and will be used for academic purposes only.

PURPOSE:

The purpose for this project is to test the predictiveness of several features measured either directly or indirectly in a seried of dynamic cycling experiments for State of Health (i.e. battery degradation). Currently, most industrial testing on battery health is done using constant current cycling. However, recent papers suggest that a dynamic cycling method (e.g. including rest intervals, etc) could reduce battery degradation. 

SOURCE DATA:

All data was pulled from this Stanford Digital Repository link:
https://purl.stanford.edu/td676xr4322

Citation: Geslin, A., Xu, L., Ganapathi, D., Moy, K., Chueh, W., and Onori, S. (2024). Dataset - Dynamic cycling enhances battery lifetime. Stanford Digital Repository. Available at https://purl.stanford.edu/td676xr4322/version/1. https://doi.org/10.25740/td676xr4322.

RAW DATA DOWNLOADING & SAVING INSTRUCTIONS:

Please keep in mind that to do the above, you must individually download all files (187 in total, 184 of which are data - 92 raw data files and 92 aging data files) locally to your computer. These files are large in size (many > 100 MB) so, in order to upload to this GitHub Repository, we utilized Git Large File Storage (Git LFS). All files associated from the Geslin et al. source are located in the Data section of the Repo. Data/AgingData contains all 92 aging files. These files include Normalized Charge Capacity, Normalized Discharge Capacity, and Normalized Cumulative Capacity at the cycle level. Data/RawData contains all 92 raw cell data files. These files include a timer (in seconds), cycle, step, and loop identification as well as voltage, current, and capacity of the cell. Within Data/Other is the README from Geslin et al. as well as publishing_data_protocol_mapping_dic.json which has a dictionary of all the cell cycling protocols that were run and which cell number they correspond to. This file was directly consulted when determining how to bin each of the cell data in later files.

RAW DATA PROCESSING:

The RawData and AgingData of each cell are processed using the struct_data.py file. This file first imports the raw data and aging data and confirms that they can be merged. The raw data is compressed to the step level (lowest level) to gain extreme attributes and durations of individual actions. These features are then compared to the rest of the data from a given cycle and both averages, maximums, and durations are generated/compressed for the overal cycle level.  The aging data is then merged with the processed raw data to create a new dataframe that contains all the processed data and aging data at the cycle level. The dataframes are then separated into the four recipe types: constant current, periodic, synthetic, and real driving. The dataframes are saved as dictionaries of dataframes (with keys identifying the recipe type) to pickle files for ease of use in subsequent python analysis files. These files are saved to Data/ProcessedData and will overwrite any same named files that were there previously. Because these files were used down the line and each test of this file generates new files, the four complete (all cycles accounted for) files were copied to Data/For_Analysis. 

As a recommendation with running the above file - please read the # NOTE on line 260. If you plan to generate the entire processed files again, keep nrows=NONE on line 261. This code will take several minutes to run through the large raw files. If you want to confirm structure or efficacy, I would recommend setting nrows=10000 and the code will complete within seconds.

DATA ANALYSIS:

The overall goal of this analysis is to see how well three different battery cycling recipe types (constant current - CC, periodic current - P, and synthetic cycling - S) predict real driving battery SOH degradation. We identified the parameter "Normalized Discharge Capacity" to represent SOH, our label. We wanted to target SOH at particular cycle numbers, so our label became a 2-D array including both variables. 

There are several files that were created to analyze these data but they are all derived from the original model_analysis.py file. For this reason, the model_analysis.py file and its generated analysis section within the Data tab will be explained in full and all subsequent files will speak to the differences from the original file. 

The model_analysis.py file first loads in all of the Processed Data locatee in Data/For_Analysis then inspects the data through inspect_dictionary and inspect_all_data for debugging purposes. Initially, it was not incorporated but the next function run is remove_outliers because some Normalized Discharge Capacity (SOH) values are uncharacteristically high which showed to have a significant impact on model fitting. A median of closely surrounding data for all features and labels is taken if the SOH is greater than a given THRESHOLD in order to muffle extreme outliers. Once the data is processed, it is ready to be analyzed. We took the approach of training and validating models of each recipe type and then testing on each individual real driving data set. That is for example -  we take all cell cycling data for continuous current, split it into a 80% train, 20% validation set using the create_train_val_sets function and then take our best model and test its prediction accuracy with the 8 real driving cells. For each train/validation set, we performed Linear, Lasso, and Ridge Regression at various hyperparameter values. For each recipe type (CC, P, or S), we identified the best performing model by calculating R^2 of the validation set when tested on the training model using the create_models function. One of the dataframes generated from this function is the coefficients_df. These were saved for each recipe type and uploaded to Data/Completed_Analysis as .csv files. These files contain model names, hyperparameters, R^2 of validation test sets, and coefficients for each feature when predicting for SOH. This file is very helpful in identifying whether a feature is overpowering other features. The identified "best models" for each recipe type were fed into the test_model function


***TO BE CONTINUED (WILL EXPLAIN ALL OTHER FILES CONCISELY generated in the repo as of 12MAR)***

