from flask import Flask, render_template, url_for, request, session, redirect, abort, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("dasboard.html")

if __name__ == "__main__":
    app.run(debug=True)
