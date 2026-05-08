from flask import Flask, render_template, url_for, request, session, flash, redirect
from datetime import timedelta

app = Flask(__name__)
app.secret_key = "Rounak2004"
app.permanent_session_lifetime = timedelta(minutes=10)

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200),nullable=False)


@app.route("/")
def home():
    if "user" in session:
        return render_template("user.html", username=session["user"])
    return render_template("register.html")

@app.route("/register", methods = ["GET","POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password =  generate_password_hash(request.form["password"])
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("Username already exists","error")
            return redirect(url_for("register"))
        
        new_user = User(username=username,password=password)
        db.session.add(new_user)
        db.session.commit()
        flash("Registered in successfully", "success")
        return redirect(url_for("login")) 
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        session.permanent = True

        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password,password):
            session["user"] = username
            flash("Logged in successfully", "success")
            return redirect(url_for("user"))
        elif not user: 
            flash("No such username register first","error")
            return redirect(url_for("register"))
        else:
            flash("Invalid username or password","error")  
    return render_template("login.html")   



@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "user" not in session:
        return redirect(url_for("login"))

    message = None

    if request.method == "POST":
        session["display_name"] = request.form.get("display_name", "").strip()
        session["bio"]          = request.form.get("bio", "").strip()
        session["email"]        = request.form.get("email", "").strip()
        message = "Profile updated successfully."

    return render_template(
        "profile.html",
        user         = session.get("user"),
        display_name = session.get("display_name", session.get("user", "")),
        bio          = session.get("bio", ""),
        email        = session.get("email", ""),
        message      = message,
    )


@app.route("/user")
def user():
    if "user" in session:
        return render_template("user.html", user=session["user"])
    else:
        flash("You are not logged in", "error")
        return redirect(url_for("login"))


@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out successfully!", "success")
    return redirect(url_for("login"))


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)

    
