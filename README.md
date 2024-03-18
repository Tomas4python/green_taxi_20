# SUBMISSION OF PRACTICAL PROJECT 4

## PROJECT - Predicting Green Taxi Fares and Total Amounts 
This project focuses on building a predictive model to estimate fare amounts and total amounts for rides taken
with green taxis in New York City. By analyzing historical trip data, I aim to develop a machine learning model
that can accurately predict the cost of a trip. The project uses scikit-learn's machine learning algorithms to
train the model on the historical trip data. 

## DATA SET
The data set used in this project contains detailed records of green taxi trips in New York City for the year 2020.
The data set can be downloaded from the following source:
[Download the 2020 NYC Green Taxi Trip Data](https://catalog.data.gov/dataset/2020-green-taxi-trip-data-january-june).
Due to the size of the dataset, it is not included in the repository.<br>
Most detailed description of columns can be found in 2022 Green Taxi Trip Data
[landing site](https://data.cityofnewyork.us/Transportation/2022-Green-Taxi-Trip-Data/8nfn-ifaj/about_data).

## PROJECT COMPONENTS
This project is divided into three Jupyter notebooks and an Excel file for model performance comparison:

1. `green_taxi_20_eda_cleanup.ipynb`: Conducts exploratory data analysis, data cleaning, and feature engineering,
resulting in a cleaned dataset saved as `cleaned_project_data_20.csv`.

2. `green_taxi_20_ml_models.ipynb`: Compares different machine learning models using the cleaned data to predict
taxi fares. Various regression algorithms are tested and evaluated to determine which model yields the most accurate
predictions.

3. `green_taxi_20_gradient_boosting.ipynb`: Focuses on fine-tuning a Gradient Boosting model for the best
performance on fare prediction. This involves hyperparameter optimization and testing different configurations
to enhance model accuracy.

4. `model_performance_metrics.xlsx`: A Excel file that documents the performance metrics of the various machine
learning models tested in the project. It includes details such as the number of rows processed, the number of
features used, hyperparameters configuration, and the final test results for comparison.

## DATA EXPLORATION AND PREPROCESSING
During the initial exploration of the dataset, I executed the following preprocessing steps to ensure data quality
and relevance for the predictive modeling:

1. Converted pickup and dropoff times to datetime format for better manipulation.
2. Removed rows with NaN values due to inconsistent or non-recoverable data.
3. Excluded non-correlated columns to streamline the dataset for fare prediction.
4. Conducted data cleansing to eliminate invalid entries:
   - Discarded trips with non-positive distances or durations.
   - Limited trip durations to a maximum of 7,200 seconds (2 hours).
   - Capped trip distances at 40 miles.
5. Identified and removed outliers:
   - Filtered out fares below the $2.5 minimum and above $150, as well as trips with durations outside the 10-second
   to 2-hour range.

These steps improved the dataset's correlations, enhancing the potential accuracy of the subsequent modeling.

## FEATURE ENGINEERING AND SELECTION
In the feature engineering phase, I enhanced the dataset with new features derived from existing data and assessed
their importance for predicting taxi fares:
1. **Trip Duration Calculation**: Generated `trip_duration` by calculating the time difference between pickup and
dropoff timestamps.
2. **Time-based Features**: Identified and added binary indicators for `night` and `peak_hour` periods, which
influence the total amount due to fare rate variations.
3. **Congestion Surcharge Indicator**: Created a binary `congestion` feature from `congestion_surcharge`.
This required careful consideration as it could potentially lead to data leakage; hence, it's a factor under
scrutiny for its actual benefit versus possible model contamination.
4. **Fare without Tips**: Formulated an `amount_without_tip` feature by subtracting tips from the total amount,
addressing inconsistencies in tip data across different payment types. <br>

Each new feature was selected based on its expected impact on the fare, with careful analysis to prevent data leakage
and ensure the robustness of the model.

## MODEL SELECTION

The second notebook is dedicated to experimenting with various regression models to identify the one that best
predicts taxi fares:

1. Linear Regression Ridge
2. K Nearest Neighbors
3. Support Vector Machines (SVM)
4. Decision Tree Regressor
5. Random Forests Regressor
6. Random Forests Regressor (with additional zone columns)
7. Random Forests Regressor (using one-hot encoding)

The Support Vector Machines (SVM) model exhibited superior performance, but due to computational constraints,
it was only feasible to train it on a subset of the data (50,000 rows). Considering the full dataset exceeds
1 million data points, this limitation necessitated setting aside the SVM approach for future exploration when
more time and resources are available. Here are several reasons why SVM might perform better for my price
prediction task: SVM can handle high dimensionality well, SVM is less sensitive to outliers than some other
algorithms and others.

## MODEL TRAINING, EVALUATION, HYPERPARAMETER TUNING
For the predictive modeling, ***Gradient Boosting*** was selected as the base algorithm. Various techniques including
GridSearchCV, RandomizedSearchCV, and manual tuning were employed to find the optimal hyperparameters. The main
challenges addressed were the presence of outliers and noise in the data, which can significantly impact model
performance.

Evaluation metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) were used on the validation set
to assess model accuracy. Data were split into training, validation, and test sets for a robust evaluation.

Several hyperparameter configurations were tested, aiming to minimize prediction errors. Notably, adjusting the `loss`
function to 'huber' and tuning the `alpha` parameter helped improve robustness against outliers, as indicated by lower
error metrics. The best-performing models featured a combination of parameters that balanced complexity with predictive
power, specifically those with higher `n_estimators` and `max_depth`, and configurations utilizing the 'huber' loss.

The results are as follows:
- The model with `n_estimators=300`, `learning_rate=0.2`, `max_depth=6`, `min_samples_split=8`, `min_samples_leaf=4`,
`loss='huber'`, and `alpha=0.95` yielded a MAE of 0.5756 and an RMSE of 2.3741, suggesting it a litle better handled the
dataset's noise and outliers.
- Models with a higher number of `min_samples_leaf` resulted in slightly higher errors, which implies a trade-off
between bias and variance.
- The use of 516 columns after encoding categorical variables, did not improve the model. <br>

In summary, the hyperparameter tuning process highlighted the importance of the 'huber' loss function and careful
selection of tree-specific parameters to mitigate the effects of noisy data and outliers on the prediction accuracy.

