from flask import Blueprint, flash, render_template, request, url_for, redirect, session
import joblib
import pandas as pd
views = Blueprint('views', __name__)

@views.route("/")
def home():
    return render_template("home.html")

@views.route("/form")
def customer_details():
    return render_template("customerDetails.html")

@views.route("/customerDetails", methods=['POST'])
def customer_details_post():
    form_data = {}

    if request.method == 'POST':
        # Add each form field to the dictionary
        form_data['Gender'] = request.form.get('Gender')
        form_data['Partner'] = request.form.get('Partner')
        form_data['Dependents'] = request.form.get('Dependents')
        form_data['PhoneService'] = request.form.get('PhoneService')
        form_data['MultipleLines'] = request.form.get('MultipleLines')
        form_data['InternetService'] = request.form.get('InternetService')
        form_data['OnlineSecurity'] = request.form.get('OnlineSecurity')
        form_data['OnlineBackup'] = request.form.get('OnlineBackup')
        form_data['DeviceProtection'] = request.form.get('DeviceProtection')
        form_data['TechSupport'] = request.form.get('TechSupport')
        form_data['StreamingTV'] = request.form.get('StreamingTV')
        form_data['StreamingMovies'] = request.form.get('StreamingMovies')
        form_data['Contract'] = request.form.get('Contract')
        form_data['PaperlessBilling'] = request.form.get('PaperlessBilling')
        form_data['PaymentMethod'] = request.form.get('PaymentMethod')
        form_data['MonthlyCharge'] = request.form.get('monthlyCharge')
        form_data['TotalCharge'] = request.form.get('totalCharge')

        try:
            with open("model.sav", "rb") as model_file:
                model = joblib.load(model_file)
                # Prepare input data for prediction
                x_df = pd.DataFrame.from_dict([form_data])

                # Apply preprocessing steps to x_df if necessary

                # Use the model to make predictions
                prediction = model.predict(x_df)

                print(f"Prediction: {prediction}")

        except FileNotFoundError:
            flash("Model file not found")
        except:
            flash("Error loading the model")

    return render_template("home.html")