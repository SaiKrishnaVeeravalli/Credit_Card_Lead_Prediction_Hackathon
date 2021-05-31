# Credit Card Lead Prediction

## Requirements
- python > 3.6
- pandas
- numpy
- matplotlib
- seaborn
- sklearn
- seaborn
- xgboost
- catboost  
- imblearn

## Data Understanding
- Shape of the data
- Number of columns and their dtypes
- Null value analysis (Found some Null values in Credit Product column which are of significant amount, so didnt remove them)

## Descriptive Analysis
- Exploratory Data Analysis is done for each independent column with target column and meaningfull insights are observed>
  * Gender - Male are more inclined for credit card.
  * Age - People from 43 to 60 years of age shown much interest for credit cards.
  * Region_Code - People from some regions have shown interest.
  * Occupation - Self_Employed and Salaried people are our lead compared to Entrepreneur.
  * Channel_Code - X2 and X3 are more intrested
  * Vintage - people with vintage 13-33 and 80-99 are mre intrested
  * Credit_Product - People who already have credit product have shown more affinity.(This column has null values and as observed 80% of people out of these null rows are                            likely to have a credit card.)
  * Is_Active - People who are not active are in need of money
  * Avg_Account_Balance - People with higher account balance have shown more interest.

## Pre-Processing
  * Label Encoding is done for columns Credit_Product, Region_Code, Gender, Occupation, Is Active, Channel_Code.(Also tried binning the data for some columns and then encode     but results are degraded)
  * Log transformation is applied for Avg_Account_Balance column since it is right skewed.
  * From heatmap we can observe the correlation between variables and target column.

## Model Building
  * Train test split is done with 80:20 ratio.
  * We can observe that there is an imbalance in our target variable. Oversampling is done to balance the classes.
  * Started with baseline XGBoostClassifier and then manually finetuned it.
  * Also used CatBoostClassifier and optimized it.
  * It is observed that both the models performed well and so I went for stacking.
  * For StackingClassifier base models XGBoost and CatBoost are passed and LogisticRegression is used are meta model.
  * Final Model is created!

## Predictions
  * Atlast predictions are made for test data and got roc auc score of 0.8732186587


