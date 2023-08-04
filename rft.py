import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd

class RF:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data, self.output_data = self.extract_data()
        self.preprocessed = self.preprocess_data(self.data)
        self.results = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1_score = None

    def preprocess_data(self, data):
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

    def extract_data(self):
        with open(self.file_name, newline='') as csvfile:
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
                        row[21],  # class
                    ]
                    data.append(row_data)
        return data, output_data

    def convert_to_gwa(self, grade):
        if grade >= 97:
            return 1.0
        elif grade >= 94:
            return 1.25
        elif grade >= 91:
            return 1.5
        elif grade >= 88:
            return 1.75
        elif grade >= 85:
            return 2.0
        elif grade >= 82:
            return 2.25
        elif grade >= 79:
            return 2.5
        elif grade >= 76:
            return 2.75
        elif grade >= 75:
            return 3.0
        else:
            return 5.0
                    

    # Define the function to convert the numerical GWA to alphanumeric grade
    def convert_to_alphanumeric(self, gwa):
        if gwa == 1.0:
            return 'A+'
        elif gwa <= 1.25:
            return 'A'
        elif gwa <= 1.5:
            return 'A-'
        elif gwa <= 1.75:
            return 'B+'
        elif gwa <= 2.0:
            return 'B'
        elif gwa <= 2.25:
            return 'B-'
        elif gwa <= 2.5:
            return 'C+'
        elif gwa <= 2.75:
            return 'C'
        elif gwa <= 3.0:
            return 'C-'
        else:
            return 'Failed'


    # Define the function to convert the numerical GWA to descriptive grade
    def convert_to_descriptive(self, gwa):
        if gwa == 1.0:
            return 'Excellent'
        elif gwa <= 1.5:
            return 'Very Good'
        elif gwa <= 2.0:
            return 'Good'
        elif gwa <= 2.75:
            return 'Fair'
        elif gwa <= 3 : #added
            return 'Passed'
        else: #failed if morethan 3
            return 'Failed'

    def train_rf_model(self):
        data = self.preprocessed
        output_data = self.output_data
        data = np.array(data)  # Convert the data list to a numpy array

        # Ensure data is 2D by adding an axis
        data = np.reshape(data, (-1, len(data[0])))

        # Encode the categorical variables (columns 0, 1, 2, 3, 5, 9, 10, 11, 13, 14)
        encoder = OneHotEncoder(handle_unknown='ignore')
        X_categorical = encoder.fit_transform(data[:, [0, 1, 2, 3, 5, 9, 10, 11, 13, 14]])
        X_numerical = data[:, 15:-2].astype(float)

        X = np.hstack((X_categorical.toarray(), X_numerical))

        # Encode the target variable (last column) to numerical values
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(data[:, -1].astype(str))

        num_folds = 5
        kf = KFold(n_splits=num_folds)

        all_predictions = []
        actual_labels = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Create and train the Random Forest classifier
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)

            # Make predictions on the test set
            predictions = rf.predict(X_test)
            all_predictions += predictions.tolist()
            actual_labels += y_test.tolist()

        data_dict_list = []
        for i in range(len(data)):
            predicted_result = label_encoder.inverse_transform([all_predictions[i]])[0]  # get the original grade
            predicted_result = float(predicted_result)  # convert string grade back to float
            predicted_result = self.convert_to_gwa(predicted_result)
            grade = self.convert_to_alphanumeric(predicted_result)
            desc = self.convert_to_descriptive(predicted_result)
            data_dict = {
                "Name": data[i][0],
                "Sex": data[i][1],
                "Age": data[i][4],
                "Year": data[i][5],
                "Family Income": output_data[i][6],
                "Family Size": output_data[i][7],
                "Parent Edu Status": output_data[i][8],
                "Ave Travel Distance": output_data[i][9],
                "G11": output_data[i][21],
                "G12": output_data[i][22],
                "Predicted Result": predicted_result,
                "Grade": grade,
                "Descriptive": desc,
                "Class": output_data[i][23]
            }
            data_dict_list.append(data_dict)
        self.results = data_dict_list

        self.accuracy = accuracy_score(actual_labels, all_predictions)
        self.precision = precision_score(actual_labels, all_predictions, average='macro', zero_division=1)
        self.recall = recall_score(actual_labels, all_predictions, average='macro')
        self.f1_score = f1_score(actual_labels, all_predictions, average='macro')
