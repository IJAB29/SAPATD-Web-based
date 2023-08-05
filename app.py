from flask import Flask, render_template, url_for, request, session, redirect, abort, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
import base64
import re
import time
import os
import ast
import uuid

#TODO: Make csv uniform when being uploaded
#TODO: Maybe migrate to mysql
#TODO: Improve UI

from mlp import MLPModel
from rft import RFModel

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

# initialize db
db = SQLAlchemy(app)

app.secret_key = "hatdog" #TODO: CHANGE

# Initialize Flask-Login to check if user if logged in
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'index'

# initialize user table in db
# naa sa pinakaubos ra nag create ng db
class User(UserMixin, db.Model):
    __tablename__ = "user"
    id = db.Column(db.String(36), primary_key=True, default=str(uuid.uuid4()))
    email = db.Column(db.String(50))
    password = db.Column(db.String(50))

# constantly check if user is logged in
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)

@login_manager.request_loader
def load_user_from_request(request):
    user_id = session.get('user_id')
    if user_id:
        return User.query.get(user_id)
    return None

# redirect to login page if the credentials do not match or not found
@login_manager.unauthorized_handler
def unauthorized_handler():
    return redirect(url_for( "login" ))

def get_results_with_grades(results_with_grades):
    descriptive_counts = {"Excellent": 0, "Very Good": 0, "Good": 0, "Fair": 0, "Pass": 0, "Failed": 0}
    for result in results_with_grades:
        descriptive_counts[result["Desc"]] += 1
    
    grade_counts = {"A+": 0, "A": 0, "A-": 0, "B+": 0, "B": 0, "B-": 0, "C+": 0, "C": 0, "C-": 0, "Failed": 0}
    for result in results_with_grades:
        grade_counts[result["Grade"]] += 1
    
    fig1, ax1 = plt.subplots()
    ax1.bar(list(descriptive_counts.keys()), list(descriptive_counts.values()))
    ax1.set_title("Descriptive Percentages")
    ax1.set_ylabel("Count")
    total_count = sum(descriptive_counts.values())
    for i, v in enumerate(descriptive_counts.values()):
        ax1.text(i, v + 1, f'{(v/total_count)*100:.1f}%', ha='center')
    
    buffer1 = BytesIO()
    fig1.savefig(buffer1, format='png')
    buffer1.seek(0)
    b64chart1 = base64.b64encode(buffer1.read()).decode('utf-8')
    descriptive_chart_url = 'data:image/png;base64,' + b64chart1
    
    fig2, ax2 = plt.subplots()
    ax2.bar(list(grade_counts.keys()), list(grade_counts.values()))
    ax2.set_title("Grade Percentages")
    ax2.set_ylabel("Count")
    total_count = sum(grade_counts.values())
    for i, v in enumerate(grade_counts.values()):
        ax2.text(i, v + 1, f'{(v/total_count)*100:.1f}%', ha='center')
    
    buffer2 = BytesIO()
    fig2.savefig(buffer2, format='png')
    buffer2.seek(0)
    b64chart2 = base64.b64encode(buffer2.read()).decode('utf-8')
    grade_chart_url = 'data:image/png;base64,' + b64chart2
    
    # Return the URLs of the saved images
    return descriptive_chart_url, grade_chart_url


ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

   
@app.route('/predict_multiceptron', methods=['GET', 'POST'])
@login_required
def multiceptron():
    if request.method == 'POST':
        start_time = time.time()
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            upload_time = time.time() - start_time

            # Select the features and target variable
            features = [
                'Dropped subject', 'Failed subject',
                '1st_year_grade_1st_sem', '1st_year_grade_2nd_sem',
                '2nd_year_grade_1st_sem', '2nd_year_grade_2nd_sem',
                '3rd_year_grade_1st_sem', '3rd_year_grade_2nd_sem'
            ]

            target = '3rd_year_grade_2nd_sem' # Adjust this to your actual target column


            # use MLP from mlp.py
            mlp_model = MLPModel()
            mlp_model.train_mlp(file_path=file_path, features=features, target=target)

            results = mlp_model.results
            results_with_grades = results.to_dict(orient="records")
            r2_mlp_percentage = mlp_model.r2_mlp_percentage
            session["r2_mlp_percentage"] = round(r2_mlp_percentage, 2)

            # Call the function to generate pie charts
            descriptive_chart_url, grade_chart_url = get_results_with_grades(results_with_grades)

            # Pass the URLs to the template
            return render_template('multiceptron.html', results_with_grades=results_with_grades, descriptive_chart_url=descriptive_chart_url, grade_chart_url=grade_chart_url, upload_time=upload_time)

        else:
            flash('Allowed file types are csv')
            return redirect(request.url)

    return render_template('multiceptron.html')
   
@app.route('/predict_randomforest', methods=['GET', 'POST'])
@login_required
def randomforest():
    if request.method == 'POST':
        start_time = time.time()
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            upload_time = time.time() - start_time

            # Select the features and target variable
            features = [
                'Dropped subject', 'Failed subject',
                '1st_year_grade_1st_sem', '1st_year_grade_2nd_sem',
                '2nd_year_grade_1st_sem', '2nd_year_grade_2nd_sem',
                '3rd_year_grade_1st_sem', '3rd_year_grade_2nd_sem'
            ]

            target = '3rd_year_grade_2nd_sem' # Adjust this to your actual target column

            # use RF
            rf_model = RFModel()
            rf_model.train_rft(file_path=file_path, features=features, target=target)

            results = rf_model.results
            results_with_grades = results.to_dict(orient="records")
            r2_rf_percentage = rf_model.r2_rf_percentage
            session["r2_rf_percentage"] = round(r2_rf_percentage, 2)

            # Call the function to generate pie charts
            descriptive_chart_url, grade_chart_url = get_results_with_grades(results_with_grades)

            # Pass the URLs to the template
            return render_template('randomforest.html', 
                                   results_with_grades=results_with_grades, 
                                   descriptive_chart_url=descriptive_chart_url, 
                                   grade_chart_url=grade_chart_url, 
                                   upload_time=upload_time)

        else:
            flash('Allowed file types are csv')
            return redirect(request.url)

    return render_template('randomforest.html')


@app.route('/piecharts_mlp', methods=['GET', 'POST'])
@login_required
def piecharts_mlp():
    if request.method == 'POST':
        descriptive_chart_url = request.form['descriptive_chart_url']
        grade_chart_url = request.form['grade_chart_url']
        
        r2_mlp_percentage = session.get("r2_mlp_percentage")

        try:
            results_with_grades = ast.literal_eval(request.form['results_with_grades'])
        except SyntaxError:
            results_with_grades = []
        return render_template('piecharts_mlp.html', 
                               descriptive_chart_url=descriptive_chart_url, grade_chart_url=grade_chart_url, 
                               results_with_grades=results_with_grades,
                               r2_mlp_percentage=r2_mlp_percentage)

    return render_template('piecharts_mlp.html')

@app.route('/piecharts_rf', methods=['GET', 'POST'])
@login_required
def piecharts_rf():
    if request.method == 'POST':
        descriptive_chart_url = request.form['descriptive_chart_url']
        grade_chart_url = request.form['grade_chart_url']

        r2_rf_percentage = session.get("r2_rf_percentage")

        try:
            results_with_grades = ast.literal_eval(request.form['results_with_grades'])
        except SyntaxError:
            results_with_grades = []
        return render_template('piecharts_rf.html', 
                               descriptive_chart_url=descriptive_chart_url, grade_chart_url=grade_chart_url, 
                               results_with_grades=results_with_grades,
                               r2_rf_percentage=r2_rf_percentage)

    return render_template('piecharts_rf.html')


@app.route("/")
@login_required
def dashboard():
    return render_template("dashboard.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()
        if user is not None and user.password == password:
            login_user(user)
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        # print(email, password)

        # Check if the email is already registered
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered. Please choose a different email.", "error")
            return redirect(url_for("register"))

        register = User(email=email, password=password)
        db.session.add(register)
        db.session.commit()
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

if __name__ == "__main__":
    # create database
    with app.app_context():
        db.create_all()
        
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.run(debug=True)
