from flask import Flask,request,render_template,jsonify
import pickle
import numpy as np
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('new_design.html')

@app.route('/predict', methods= ['POST'])
def predict():

    input_data = request.form
    # print(input_data)
    
    with open(r'artifacts\feature_names.json','r') as file:
        feature_name= json.load(file)
    
    Gender = int(input_data['Gender'])
    Married = int(input_data['Married'])
    Dependents = int(input_data['Dependents'])
    Education = int(input_data['Education'])
    Self_Employed = int(input_data['Self_Employed'])
    Loan_Amount_Term = float(input_data['loan_term'])
    Credit_History = float(input_data['Credit_History'])
    Property_Area = int(input_data['Property_Area'])
    
    LoanAmount = int(input_data['loan_amount'])
    # Converting in log due to preprocessing on training data 
    log_loan_amount = np.log(LoanAmount)
    
    applicant_income = int(input_data['applicant_income'])
    coapplicant_income = int(input_data['coapplicant_income'])
        # Converting in log due to preprocessing on training data
    total_income = applicant_income + coapplicant_income
    log_total_income = np.log(total_income)


    arr = np.array([Gender,Married,Dependents,Education,Self_Employed,Loan_Amount_Term,Credit_History,
    Property_Area,log_loan_amount,log_total_income])
    print(arr)

    

    # Loadning Model 
    with open(r'artifacts\model.pkl','rb') as file:
        model = pickle.load(file)
    ## Predicting the results
    result = model.predict([arr])
    print(result)
    if result[0] == 1:
        result = "Loan Approved"
    else: 
        result = "Loan Rejected"

    return render_template('index.html', prediction = result)

if __name__ == "__main__":
    app.run(debug=True, port=5000,host='localhost')

