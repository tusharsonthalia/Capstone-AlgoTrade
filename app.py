import os
import json
import pandas as pd

from cs50 import SQL
from flask import Flask, flash, jsonify, redirect, render_template, request, session
from flask_session import Session
from tempfile import mkdtemp
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
from werkzeug.security import check_password_hash, generate_password_hash

from helpers import apology, login_required, lookup, usd
from deep_learning import DeepLearner

with open('config.json') as f:
    config = json.load(f)


# Configure application
app = Flask(__name__)

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


# Custom filter
app.jinja_env.filters["usd"] = usd

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Configure CS50 Library to use SQLite database
db = SQL("sqlite:///finance.db")


@app.route("/", methods=['GET', 'POST'])
@login_required
def index():
    return render_template("index.html")

@app.route("/deep_learning", methods=['GET', 'POST'])
@login_required
def deep_learning():
    if request.method == "GET":
        algorithms = config['Deep Learning']
        stocks = pd.read_csv("data/stock_list.csv")['Symbol'].tolist()

        return render_template("deep_learning.html", algorithms=algorithms, stocks=stocks)
    
    else: 
        algorithm = request.form.get('nameAlgo')
        stock = request.form.get('nameStock')
        retrain = request.form.get('retrainModel')
        train = request.form.get('trainDays')
        predict = request.form.get('predictDays')
        lag = request.form.get('lagDays')
        
        # input validation
        if algorithm == 'null': algorithm = 'Lstm'
        if stock == 'null': stock = 'ACC'
        try: 
            train = int(train)
        except ValueError:
            train = 30
        try: 
            predict = int(predict)
        except ValueError:
            predict = 1
        try: 
            lag = int(lag)
        except ValueError:
            lag = 0
        if retrain == 'Yes': retrain = True
        else: retrain = False

        predictions = DeepLearner(stock, algorithm, retrain, train, predict, lag).predict()
        
        return render_template("deep_learning_output.html", algorithm=algorithm,
                                stock=stock, train=train, predict=predict, lag=lag,
                                predictions=predictions, retrain=False)




@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Ensure username was submitted
        if not request.form.get("username"):
            return apology("must provide username", 403)

        # Ensure password was submitted
        elif not request.form.get("password"):
            return apology("must provide password", 403)

        # Query database for username
        rows = db.execute("SELECT * FROM users WHERE username = :username",
                          username=request.form.get("username"))

        # Ensure username exists and password is correct
        if len(rows) != 1 or not check_password_hash(rows[0]["hash"], request.form.get("password")):
            return apology("invalid username and/or password", 403)

        # Remember which user has logged in
        session["user_id"] = rows[0]["id"]

        # Redirect user to home page
        return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")

@app.route("/logout")
def logout():
    """Log user out"""

    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/")



@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""
    # handling the case of a new user registering
    if request.method == 'POST':

        # storing the form data in variables
        username = request.form.get("username")
        password = request.form.get("password")
        confirm_password = request.form.get("confirmation")

        # checking if the username is invalid
        if not username:
            return apology("Must Provide Username.")

        # checking if the password field is empty
        elif not password:
            return apology("Must Provide Password.")

        # checking if the confirm password field is empty
        elif not confirm_password:
            return apology("Must Confirm Password.")

        # checking if the passwords match or not
        if password != confirm_password:
            return apology("The passwords in the password field and the confirm \
                password field do not match. Please try again.")

        # checking if the username is unique
        rows = db.execute("SELECT * FROM users WHERE username=:username", username=username)
        if len(rows) > 0:
            return apology("Please use a unique username")

        # hashing the password
        password = generate_password_hash(password)

        # inserting the user into the database
        db.execute("INSERT INTO users (username, hash) VALUES (:username, :password_hash)",
                   username=username, password_hash=password)

        # remembering the user id and logging the user in
        rows = db.execute("SELECT * FROM users WHERE username=:username AND hash=:password",
                          username=username, password=password)
        session["user_id"] = rows[0]["id"]

        # redirecting the user to the index page
        return redirect('/')

    # handling the case of a user registering
    else:
        return render_template("register.html")

def errorhandler(e):
    """Handle error"""
    if not isinstance(e, HTTPException):
        e = InternalServerError()
    return apology(e.name, e.code)

# Listen for errors
for code in default_exceptions:
    app.errorhandler(code)(errorhandler)

app.run(debug=True)