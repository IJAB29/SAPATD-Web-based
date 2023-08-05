import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def convert_to_alphanumeric(gwa):
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


def convert_to_descriptive(gwa):
    if gwa == 1.0:
        return 'Excellent'
    elif gwa <= 1.5:
        return 'Very Good'
    elif gwa <= 2.0:
        return 'Good'
    elif gwa <= 2.75:
        return 'Fair'
    else:
        return 'Pass'
    
def round_to_grade(predicted_value):
    grades = [1.0, 1.25, 1.50, 1.75, 2.0, 2.25, 2.50, 2.75, 3.0, 5.0]
    return min(grades, key=lambda x: abs(x - predicted_value))
    

def preprocess_data(X):
    # Handle missing values
    X.fillna(X.mean(), inplace=True)

    # Standardize the features
    scaler_X = StandardScaler().fit(X)
    X_scaled = scaler_X.transform(X)

    return X_scaled

# # Read the dataset
# file_path = "dummy_data.csv" # Change to your CSV file path

# # Select the features and target variable
# features = [
#     'Dropped subject', 'Failed subject',
#     '1st_year_grade_1st_sem', '1st_year_grade_2nd_sem',
#     '2nd_year_grade_1st_sem', '2nd_year_grade_2nd_sem',
#     '3rd_year_grade_1st_sem', '3rd_year_grade_2nd_sem'
# ]

# target = '3rd_year_grade_2nd_sem' # Adjust this to your actual target column

class RFModel:
    def __init__(self):
        self.r2_rf_percentage = None
        self.results = None

    def train_rft(self, file_path, features: list, target: str):
        df = pd.read_csv(file_path)

        X = df[features]
        y = df[target]



        # Preprocess the features
        X_scaled = preprocess_data(X)

        # Train the Random Forest model (changed from MLP)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)

        # Predict on the entire dataset
        y_pred_all = rf.predict(X_scaled)

        # Round the predictions to the nearest valid grade
        rounded_predictions = [round_to_grade(p) for p in y_pred_all]

        # Compute the mean of the target variable (this is the benchmark prediction)
        # Mean target value
        mean_target = y.mean()

        # Baseline predictions (always the mean target)
        baseline_predictions = [mean_target] * len(y)

        # MLP model absolute errors
        absolute_model_error = abs(y.reset_index(drop=True) - rounded_predictions)

        # Baseline absolute errors
        absolute_baseline_error = abs(y.reset_index(drop=True) - baseline_predictions)

        # MLP model percentage errors
        percentage_model_error = absolute_model_error / y.reset_index(drop=True) * 100

        # Baseline percentage errors
        percentage_baseline_error = absolute_baseline_error / y.reset_index(drop=True) * 100

        # Improvement in absolute errors
        improvement_absolute = absolute_baseline_error - absolute_model_error

        # Improvement in percentage errors
        improvement_percentage = percentage_baseline_error - percentage_model_error

        # Create a Pandas Series from the rounded_predictions list
        rounded_predictions_series = pd.Series(rounded_predictions)

        # Use the map function with your conversion function
        alphanumeric_grades = rounded_predictions_series.map(convert_to_alphanumeric)
        desc_grades = rounded_predictions_series.map(convert_to_descriptive)


        results_all_df = pd.DataFrame({
            'Name': df['Name'].reset_index(drop=True),
            'Gender': df['Gender'].reset_index(drop=True),
            'Age': df['Age'].reset_index(drop=True),
            'School year': df['School year'].reset_index(drop=True),
            'G11_grade': df['G11_grade'].reset_index(drop=True),
            'G12_grade': df['G12_grade'].reset_index(drop=True),
            'Actual Grade': y.reset_index(drop=True),
            'Predicted  Grade': rounded_predictions,
            'Grade': alphanumeric_grades,
            'Desc': desc_grades,
            'Improvement (Absolute)': round(improvement_absolute, 2),
            'Improvement (Percentage)': round(improvement_percentage, 2)
        })

        r2_rf = r2_score(y, y_pred_all)
        self.r2_rf_percentage = r2_rf * 100
        self.results = results_all_df


# pd.set_option('display.max_rows', None)

# print(results_all_df)
# print(f"R^2 score for the Random Forest model: {r2_rf_percentage:.2f}%")

