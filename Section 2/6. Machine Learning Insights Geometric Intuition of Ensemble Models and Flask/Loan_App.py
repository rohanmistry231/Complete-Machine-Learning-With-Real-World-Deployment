import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

#Creating the flask app
app = Flask(__name__)

model = joblib.load("Loan_Predictor.pkl")

@app.route('/')
def home():
    return render_template('Loan_App_Template.html')

@app.route('/predict', methods=['POST'])
def predict():
  
     if request.method == 'POST':
            Male=request.form['genderBox']
            if(Male=='male'):
                Male=1
            else:
                Male=0

            Marital_Status = request.form['maritalBox']
            if(Marital_Status == 'Married'):
                Marital_Status = 1
            else:
                Marital_Status = 0	

            dependents = request.form['dependentBox']

            education = request.form['educationBox']
            if(education == 'Graduate'):
                education = 1
            else:
                education = 0

            se = request.form['employmentBackground']
            if(se == 'Self-Employed'):
                se = 1
            else:
                se = 0
                
            applicantsIncome = request.form['applicantIncomeBox']

            coApplicantsIncome = request.form['coApplicantIncomeBox']

            loanAmt = request.form['laonAmtBox']

            loanTerm = request.form['laonAmtTermBox']

            ch = request.form['CHBox']
            if(ch == 'Good'):
                ch = 1
            else:
                ch = 0

            propArea = request.form['propertyAreaBox']
            if(propArea == 'Rural'):
                Rural = 1
                SemiUrban = 0
            elif(propArea == 'Semi Urban'):
                Rural = 0
                SemiUrban = 1 
            else:
                Rural=0
                SemiUrban=0 

            prediction = model.predict([[Male, Marital_Status, dependents, education, se, applicantsIncome, 
            coApplicantsIncome, loanAmt, loanTerm, ch, Rural, SemiUrban]])

            if (prediction[0] > 0.5):
                return render_template('Loan_App_Template.html', prediction_text='RESULTS:  Loan Approved')

            else:
                return render_template('Loan_App_Template.html', prediction_text= 'RESULTS:  Loan Rejected')

if __name__ == "__main__":
    app.run(debug=True)