# income_classification

This project is to build machine learning models to perform classification task that determines whether a person makes over 50K a year from UCI's Census Income dataset. Logisic regression and random forest models are developed for this project and the process includes data preprocessing and data cleaning, feature engineering, model training and testing, and hyperparameter tuning using grid search with cross validation. An API endpoint was set up using Flask to generate prediction result from the best ML model. The endpoint takes the input data from the test dataset (single row of data or a batch) and returns the predictions. 

The feature space is as follows, and contains a mixture of numerical and categorical inputs. 
1.	age: continuous.
2.	workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
3.	fnlwgt: continuous.
4.	education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
5.	education-num: continuous.
6.	marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
7.	occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
8.	relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
9.	race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
10.	sex: Female, Male.
11.	capital-gain: continuous.
12.	capital-loss: continuous.
13.	hours-per-week: continuous.
14.	native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.


You can find the following files in the project folder:
1. Training and test dataset (adult_trdata.csv and adult_test.csv)
2. Two ipython notebooks for ML model development (Logistic_Regression.ypynb and Random_Forest.ipynb)
3. Model output from the best ML model (logr.pkl)
4. Python file for creating API endpoint using Flask (app.py)
5. Instructions on how to test API endpoint (API_Testing_Instruction.docx)
6. Sample input data (single row of data or a batch) for API testing (test_df1.csv and test_df2.csv)
