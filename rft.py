import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder 

file_name = "knn-Test2.csv"

def preprocess_data(data):
    preprocessed_data = []
    for row in data:
        preprocessed_row = []
        for i, value in enumerate(row):
            if i in [0, 1, 2, 3, 5, 9, 10, 11, 13, 14, 22, 23, 24]:
                preprocessed_row.append(value)
            else:
                try:
                    preprocessed_row.append(float(value))
                except ValueError:
                    preprocessed_row.append(value)
        preprocessed_data.append(preprocessed_row)
    return preprocessed_data

with open(file_name, newline='') as csvfile:
    datareader = csv.reader(csvfile)
    data = []
    output_data = []
    next(datareader) # skip head
    for row in datareader:
        if len(row) >= 23:
            output_data.append(row)
            row_data = [
                row[0],  # name
                row[1],  # sex
                row[2],  # nationality
                row[3],  # birthplace
                int(row[4]),  # age
                row[5],  # year
                row[6],  # family_income
                row[7],  # family_size
                row[8],  # parent_educ_status
                row[9],  # ave_travel_res_schl
                int(row[10]),  # study_hours
                row[11],  # studs_hour
                int(row[12]),  # dropped_subjects
                row[13],  # failed_subjects
                row[14],  # love_of_course
                float(row[15]),  # 1st_yr_1st_sem_gwa
                float(row[16]),  # 1st_yr_2nd_sem_gwa
                float(row[17]),  # 2nd_yr_1st_sem_gwa
                float(row[18]),  # 2nd_yr_2nd_sem_gwa
                float(row[19]),  # 3rd_yr_1st_sem_gwa
                float(row[20]),  # 3rd_yr_2nd_sem_gwa
                row[23],  # class
            ]
            data.append(row_data)

# Define the function to convert the numerical GWA to alphanumeric grade

def convert_to_alphanumeric(gwa_class):
    return gwa_class

# Define the function to convert the numerical GWA to descriptive grade
def convert_to_descriptive(gwa_class):
    return gwa_class

# ... (rest of the code remains the same)

data = preprocess_data(data)
data = np.array(data)  # Convert the data list to a numpy array

# Ensure data is 2D by adding an axis
data = np.reshape(data, (-1, len(data[0])))

# Split the features (X) and target (y)
X_categorical = data[:, [0, 1, 2, 3, 5, 9, 10, 11, 13, 14]]  # Categorical features
X_numerical = data[:, 15:-1].astype(float)  # Numerical features
X = np.hstack((X_categorical, X_numerical))  # Combined features
y = data[:, -1]  # Class column

num_folds = 5
kf = KFold(n_splits=num_folds)

all_predictions = []
actual_labels = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Encode the categorical variables using OneHotEncoder
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_train_categorical = encoder.fit_transform(X_train[:, :10])
    X_test_categorical = encoder.transform(X_test[:, :10])

    # Concatenate the categorical and numerical features
    X_train_encoded = np.hstack((X_train_categorical.toarray(), X_train[:, 10:].astype(float)))
    X_test_encoded = np.hstack((X_test_categorical.toarray(), X_test[:, 10:].astype(float)))

    # Create and train the Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_encoded, y_train)

    # Make predictions on the test set
    predictions = rf.predict(X_test_encoded)
    all_predictions += predictions.tolist()
    actual_labels += y_test.tolist()

for i in range(len(data)):
    predicted_result = all_predictions[i]
    print(
        f"Name: {data[i][0]}, Sex: {data[i][1]}, Age: {data[i][4]}, Year: {data[i][5]}, Family Income: {output_data[i][6]}, "
        f"Family Size: {output_data[i][7]}, Parent Edu Status: {output_data[i][8]}, Ave Travel Distance: {output_data[i][9]}, "
        f"G11: {output_data[i][21]}, G12: {output_data[i][22]}, Predicted Result: {predicted_result}, Class: {output_data[i][23]}")

accuracy = accuracy_score(actual_labels, all_predictions)
precision = precision_score(actual_labels, all_predictions, average='macro', zero_division=1)
recall = recall_score(actual_labels, all_predictions, average='macro')
f_score = f1_score(actual_labels, all_predictions, average='macro')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-score: {f_score}")