from flask import Flask, request, jsonify
import pickle
import copy
import pandas as pd
import numpy as np
from sklearn import preprocessing


app = Flask(__name__)

# load the model
MODEL = pickle.load(open("logr.pkl", "rb"))
# create model labels
MODEL_lABELS = ['<=50k','>50k']

# create webpage for uploading csv file
@app.route('/')
def form():
    return """
        <html>
            <body>
                <p> Please upload the input dataset (CSV file with single or multiple rows) </p>
                <form action="/process" method="post" enctype="multipart/form-data">
                    <input type="file" name="data_file" class="btn btn-block"/>
                    </br>
                    </br>
                    <button type="submit" class="btn btn-primary btn-block btn-large">Predict</button>
                </form>
            </body>
        </html>
    """

@app.route('/process', methods=["POST"])
def process():
    if request.method == 'POST':
        file = request.files.get('data_file')
        if file:
            input_df = pd.read_csv(file,skipinitialspace=True)
        else:
            return {"error" : "Input dataset CSV file not uploaded!"}

    # transform input dataframe into model-friendly features
    x_test = transform(input_df)

    # return prediction associated with each row of the test dataset
    res = response(input_df, x_test)
    return jsonify(res)

def response(input_df, x_test):
    res = []
    for index, rows in input_df.iterrows():
        # get feature values for each row
        features = x_test.iloc[index].values.reshape(1,-1)

        # from the model building process, rows with empty values in occupation were dropped
        # therefore return "not valid" if occupation is NaN
        occupation = rows[8]
        if occupation != "?":
            prediction = predict(features)
        else:
            prediction = "not valid"

        # prepare output format, include original columns and prediction result for each row
        data = {
            "age": rows[2],
            "workclass": rows[3],
            "fnlwgt": rows[4],
            "education": rows[5],
            "education-num": rows[6],
            "marital-status": rows[7],
            "occupation": rows[8],
            "relationship": rows[9],
            "race": rows[10],
            "sex": rows[11],
            "capital-gain": rows[12],
            "capital-loss": rows[13],
            "hours-per-week": rows[14],
            "native-country": rows[15],
            "income": rows[16],
            "prediction": prediction
        }
        res.append(data)
    return res

def transform(input_df):
    # reformat input dataframe
    input_df.reset_index(inplace=True)
    input_df.rename(columns = {"level_0":"age", "level_1":"workclass", "level_2":"fnlwgt", "level_3":"education", "level_4":"education-num",
                    "level_5":"marital-status","level_6":"occupation", "level_7":"relationship", "level_8":"race",
                    "level_9":"sex", "level_10":"capital-gain", "level_11":"capital-loss", "level_12":"hours-per-week", "level_13":"native-country",
                    "|1x3 Cross validator":"income"},inplace = True)
    test_df = copy.deepcopy(input_df)
    # convert "?" into NaN and count the number of rows that contain NaN in any column
    test_df.replace({"?":np.nan}, inplace=True)

    # fill the missing values from workclass column with "Private"
    test_df['workclass'].fillna("Private", inplace=True)

    # replace missing values from native-country with United States
    test_df['native-country'].fillna("United-States", inplace=True)

    # reduce the number of categories of 'native-country' column and reclassify based on continent
    test_df['native-country'].replace(
        to_replace=["Philippines", "India", "China", "South", "Vietnam", "Japan", "Iran", "Taiwan", "Hong", "Cambodia",
                    "Thailand", "Laos", "Outlying-US(Guam-USVI-etc)"], value="Asia", inplace=True)
    test_df['native-country'].replace(
        to_replace=["Germany", "England", "Italy", "Poland", "Portugal", "Greece", "France", "Ireland", "Yugoslavia",
                    "Hungary", "Scotland", "Holand-Netherlands"], value="Europe", inplace=True)
    test_df['native-country'].replace(
        to_replace=["Mexico", "Puerto-Rico", "El-Salvador", "Cuba", "Jamaica", "Dominican-Republic", "Guatemala",
                    "Columbia", "Haiti", "Nicaragua", "Peru", "Ecuador", "Trinadad&Tobago", "Honduras"],
        value="Latin America", inplace=True)

    # reduce the number of categories of education column
    test_df['education'].replace(to_replace=["1st-4th", "5th-6th"], value="Elementary", inplace=True)
    test_df['education'].replace(to_replace=["7th-8th", "9th", "10th", "11th", "12th"], value="Some-HS", inplace=True)

    # Discretize capital gain feature into 5 bins in test_df
    D = test_df[['capital-gain']]
    est3 = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans').fit(D)
    Dt = est3.transform(D)
    test_df['capital-gain-discretized'] = Dt

    # Discretize capital loss feature into 5 bins in test_df
    E = test_df[['capital-loss']]
    est4 = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans').fit(E)
    Et = est4.transform(E)
    test_df['capital-loss-discretized'] = Et

    # Encode target labels with values 0 and 1 in test_df
    F = test_df[['income']]
    le = preprocessing.LabelEncoder().fit(F)
    Ft = le.transform(F)
    test_df['income_encoded'] = Ft

    # Replace categorical features with one-hot-encoded columns in test_df
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                            'native-country']
    for col in categorical_features:
        col_ohe = pd.get_dummies(test_df[col], prefix=col)
        test_df = pd.concat((test_df, col_ohe), axis=1).drop(col, axis=1).drop(columns=col_ohe.columns[0], axis=1)

    # drop useless features in test_df
    test_df.drop(['capital-gain', 'capital-loss', 'income', 'fnlwgt'], axis=1, inplace=True)

    X_test = test_df.drop(['income_encoded'], axis=1)
    y_test = test_df['income_encoded']

    # all feature columns from trained logistic regression model
    all_features = ['age', 'education-num', 'hours-per-week', 'capital-gain-discretized',
       'capital-loss-discretized', 'workclass_Local-gov', 'workclass_Private',
       'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc',
       'workclass_State-gov', 'workclass_Without-pay', 'education_Assoc-voc',
       'education_Bachelors', 'education_Doctorate', 'education_Elementary',
       'education_HS-grad', 'education_Masters', 'education_Preschool',
       'education_Prof-school', 'education_Some-HS', 'education_Some-college',
       'marital-status_Married-AF-spouse', 'marital-status_Married-civ-spouse',
       'marital-status_Married-spouse-absent', 'marital-status_Never-married',
       'marital-status_Separated', 'marital-status_Widowed',
       'occupation_Armed-Forces', 'occupation_Craft-repair',
       'occupation_Exec-managerial', 'occupation_Farming-fishing',
       'occupation_Handlers-cleaners', 'occupation_Machine-op-inspct',
       'occupation_Other-service', 'occupation_Priv-house-serv',
       'occupation_Prof-specialty', 'occupation_Protective-serv',
       'occupation_Sales', 'occupation_Tech-support',
       'occupation_Transport-moving', 'relationship_Not-in-family',
       'relationship_Other-relative', 'relationship_Own-child',
       'relationship_Unmarried', 'relationship_Wife',
       'race_Asian-Pac-Islander', 'race_Black', 'race_Other', 'race_White',
       'sex_Male', 'native-country_Canada', 'native-country_Europe',
       'native-country_Latin America', 'native-country_United-States']

    # for missing columns from input test dataset, add those columns with 0 value
    for col in all_features:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test.drop(columns=['index', 'Unnamed: 0'], inplace=True)
    return X_test

def predict(features):
    # Make predictions using the model
    label_index = MODEL.predict(features)
    # Retrieve the label name that is associated with the label index
    label = MODEL_lABELS[label_index[0]]
    return label

if __name__ == "__main__":
    app.run()