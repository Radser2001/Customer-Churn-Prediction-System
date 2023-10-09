from flask import Blueprint, flash, render_template, request, url_for, redirect, session


views = Blueprint('views', __name__)

@views.route("/")
def home():
 
    return render_template("home.html")

@views.route("/customerDetails")
def customer_details():

    return render_template("customerDetails.html")
