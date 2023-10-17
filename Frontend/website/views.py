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
        gender = request.form.get('Gender')

        if gender == "Female":
            form_data['gender'] = 0
        else:
            form_data['gender'] = 1

        senior_citizen = request.form.get('SeniorCitizen')
        if senior_citizen == "Yes":
            form_data['SeniorCitizen'] = 1
        else:
            form_data['SeniorCitizen'] = 0

        partner = request.form.get('Partner')
        if partner == "Yes":
            form_data['Partner'] = 1
        else:
            form_data['Partner'] = 0

        dependents = request.form.get('Dependents')
        if dependents == "Yes":
            form_data['Dependents'] = 1
        else:
            form_data['Dependents'] = 0

        form_data['tenure'] = request.form.get('tenure')
        phone_service = request.form.get('PhoneService')
        if phone_service == "Yes":
            form_data['PhoneService'] = 1
        else:
            form_data['PhoneService'] = 0

        multiple_lines = request.form.get('MultipleLines')
        if multiple_lines == "Yes":
            form_data['MultipleLines'] = 1
        else:
            form_data['MultipleLines'] = 0

        online_security = request.form.get('OnlineSecurity')
        if online_security == "Yes":
            form_data['OnlineSecurity'] = 1
        else:
            form_data['OnlineSecurity'] = 0

        online_backup = request.form.get('OnlineBackup')
        if online_backup == "Yes":
            form_data['OnlineBackup'] = 1
        else:
            form_data['OnlineBackup'] = 0

        device_protection = request.form.get('DeviceProtection')
        if device_protection == "Yes":
            form_data['DeviceProtection'] = 1
        else:
            form_data['DeviceProtection'] = 0

        tech_support = request.form.get('TechSupport')
        if tech_support == "Yes":
            form_data['TechSupport'] = 1
        else:
            form_data['TechSupport'] = 0

        streaming_tv = request.form.get('StreamingTV')
        if streaming_tv == "Yes":
            form_data['StreamingTV'] = 1
        else:
            form_data['StreamingTV'] = 0

        streaming_movies = request.form.get('StreamingMovies')
        if streaming_movies == "Yes":
            form_data['StreamingMovies'] = 1
        else:
            form_data['StreamingMovies'] = 0

        paperless_billing = request.form.get('PaperlessBilling')
        if paperless_billing == "Yes":
            form_data['PaperlessBilling'] = 1
        else:
            form_data['PaperlessBilling'] = 0

        form_data['MonthlyCharges'] = request.form.get('monthlyCharge')
        
        form_data['TotalCharges'] = request.form.get('totalCharge')

        internet_service = request.form.get('InternetService')
        if internet_service == "DSL":
            form_data['InternetService_DSL'] = 1
            form_data['InternetService_Fiber_optic'] = 0
            form_data['InternetService_No'] = 0
        
        if internet_service == "Fiber optic":
            form_data['InternetService_DSL'] = 0
            form_data['InternetService_Fiber_optic'] = 1
            form_data['InternetService_No'] = 0

        if internet_service == "No":
            form_data['InternetService_DSL'] = 0
            form_data['InternetService_Fiber_optic'] = 0
            form_data['InternetService_No'] = 1


        contract = request.form.get('Contract')
        if contract == "One year":
            form_data['Contract_Month_to_month'] = 0
            form_data['Contract_One_year'] = 1
            form_data['Contract_Two_year'] = 0

        if contract == "Month-to-month":
            form_data['Contract_Month_to_month'] = 1
            form_data['Contract_One_year'] = 0
            form_data['Contract_Two_year'] = 0


        if contract == "Two year":
            form_data['Contract_Month_to_month'] = 0
            form_data['Contract_One_year'] = 0
            form_data['Contract_Two_year'] = 1



        
        payment_method = request.form.get('PaymentMethod')
        print(payment_method)
        if payment_method == "Bank transfer (automatic)":
            form_data['PaymentMethod_Bank_transfer__automatic_'] = 1
            form_data['PaymentMethod_Credit_card__automatic_'] = 0
            form_data['PaymentMethod_Electronic_check'] = 0
            form_data['PaymentMethod_Mailed_check'] = 0

        if payment_method == "Credit card (automatic)":
            form_data['PaymentMethod_Bank_transfer__automatic_'] = 0
            form_data['PaymentMethod_Credit_card__automatic_'] = 1
            form_data['PaymentMethod_Electronic_check'] = 0
            form_data['PaymentMethod_Mailed_check'] = 0

        if payment_method == "Electronic Check":
            form_data['PaymentMethod_Bank_transfer__automatic_'] = 0
            form_data['PaymentMethod_Credit_card__automatic_'] = 0
            form_data['PaymentMethod_Electronic_check'] = 1
            form_data['PaymentMethod_Mailed_check'] = 0

        if payment_method == "Mailed Check":
            form_data['PaymentMethod_Bank_transfer__automatic_'] = 0
            form_data['PaymentMethod_Credit_card__automatic_'] = 0
            form_data['PaymentMethod_Electronic_check'] = 0
            form_data['PaymentMethod_Mailed_check'] = 1


        with open("model.sav", "rb") as model_file:
            model = joblib.load(model_file)
                # Prepare input data for prediction
            x_df = pd.DataFrame.from_dict([form_data])

               

                # Use the model to make predictions
            prediction = model.predict(x_df)
      
            print(f"Prediction: {prediction[0]}")
    return render_template("prediction.html", prediction=prediction[0])