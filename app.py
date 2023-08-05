from flask import Flask, render_template, url_for, request, session, redirect, abort, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from io import BytesIO
import matplotlib.pyplot as plt
import base64
import re
import time
import os
import ast
import uuid

#TODO: Make csv uniform when being uploaded
#TODO: Maybe migrate to mysql
#TODO: Improve UI

from mlp import MLP
from rft import RF

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

app.secret_key = "hatdog" #TODO: CHANGE

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'index'

class User(UserMixin, db.Model):
    __tablename__ = "user"
    id = db.Column(db.String(36), primary_key=True, default=str(uuid.uuid4()))
    email = db.Column(db.String(50))
    password = db.Column(db.String(50))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)

@login_manager.request_loader
def load_user_from_request(request):
    user_id = session.get('user_id')
    if user_id:
        return User.query.get(user_id)
    return None

@login_manager.unauthorized_handler
def unauthorized_handler():
    return 'Unauthorized', 401

def get_results_with_grades(results_with_grades):
    descriptive_counts = {"Excellent": 0, "Very Good": 0, "Good": 0, "Fair": 0, "Pass": 0, "Failed": 0}
    for result in results_with_grades:
        descriptive_counts[result["Descriptive"]] += 1
    
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

            # use MLP
            mlp_model = MLP(file_name=file_path)
            mlp_model.train_mlp_model()

            results_with_grades = mlp_model.results

            print(f"Accuracy: {mlp_model.accuracy}")
            print(f"Precision: {mlp_model.precision}")
            print(f"Recall: {mlp_model.recall}")
            print(f"F-score: {mlp_model.f1_score}")

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

            # use RF
            rf_model = RF(file_name=file_path)
            rf_model.train_rf_model()

            results_with_grades = rf_model.results

            print(f"Accuracy: {rf_model.accuracy}")
            print(f"Precision: {rf_model.precision}")
            print(f"Recall: {rf_model.recall}")
            print(f"F-score: {rf_model.f1_score}")

            # Call the function to generate pie charts
            descriptive_chart_url, grade_chart_url = get_results_with_grades(results_with_grades)

            # Pass the URLs to the template
            return render_template('randomforest.html', results_with_grades=results_with_grades, descriptive_chart_url=descriptive_chart_url, grade_chart_url=grade_chart_url, upload_time=upload_time)

        else:
            flash('Allowed file types are csv')
            return redirect(request.url)

    return render_template('randomforest.html')


@app.route('/piecharts2', methods=['GET', 'POST'])
@login_required
def piecharts2():
    if request.method == 'POST':
        descriptive_chart_url = request.form['descriptive_chart_url']
        grade_chart_url = request.form['grade_chart_url']
        try:
            results_with_grades = ast.literal_eval(request.form['results_with_grades'])
        except SyntaxError:
            results_with_grades = []
        return render_template('piecharts2.html', descriptive_chart_url=descriptive_chart_url, grade_chart_url=grade_chart_url, results_with_grades=results_with_grades)

    return render_template('piecharts2.html')


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")

@app.route("/logout")
@login_required
def logout():
    # Use the logout_user function to log the user out
    logout_user()
    return redirect(url_for("index"))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()
        login_user(user)
        return redirect(url_for("dashboard"))
        # if user is not None and check_password_hash(user.password, password):
        #     # Use the login_user function to log the user in
        #     login_user(user)
        #     return redirect(url_for("dashboard"))
        # else:
        #     flash("Invalid email or password", "error")
        #     print("Invalid email or password", "error")

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
        return redirect(url_for("index"))

    return render_template("register.html")

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.run(debug=True)
