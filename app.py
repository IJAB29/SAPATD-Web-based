from flask import Flask, render_template, url_for, request, session, redirect, abort, jsonify, flash
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

app.secret_key = "hatdog"

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(50))
    password = db.Column(db.String(50))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        
        login = User.query.filter_by(email=email, password=password).first()
        if login is not None:
            return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        print(email, password)

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
    app.run(debug=True)
