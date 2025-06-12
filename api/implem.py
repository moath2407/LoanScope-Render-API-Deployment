from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
import pickle
import pandas as pd
import numpy as np

#Loading all the pickled assets
with open('model_features.pkl', 'rb') as f:
    model_features = pickle.load(f)

with open('model_Linearclassifier.pkl','rb') as f:
    LinearSVC = pickle.load(f)

with open('model_regressor.pkl','rb') as f:
    regressor = pickle.load(f)

with open('model_scaler.pkl','rb') as f:
    scaler = pickle.load(f)



app = FastAPI()

class Loan(BaseModel):
    AnnualIncome: float
    CreditScore: float
    EmploymentStatus: str
    EducationLevel: str
    Experience: float
    LoanAmount: float
    LoanDuration: float
    MaritalStatus: str
    NumberOfDependents: int
    HomeOwnershipStatus: str
    MonthlyDebtPayments: float
    CreditCardUtilizationRate: float
    NumberOfOpenCreditLines: int
    NumberOfCreditInquiries: int
    DebtToIncomeRatio: float
    BankruptcyHistory: int
    LoanPurpose: str
    PreviousLoanDefaults: int
    PaymentHistory: float
    NetWorth: float
    MonthlyLoanPayment: float
    TotalDebtToIncomeRatio: float
    TotalLiabilities: float

numerical_cols = [
    'AnnualIncome','CreditScore','Experience','LoanAmount','LoanDuration',
    'NumberOfDependents','MonthlyDebtPayments','CreditCardUtilizationRate','NumberOfOpenCreditLines',
    'NumberOfCreditInquiries','DebtToIncomeRatio','NetWorth',
    'MonthlyLoanPayment', 'TotalDebtToIncomeRatio', 'TotalLiabilities' ]

categorical_var = ['EmploymentStatus', 'EducationLevel', 'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose']
@app.get("/")
def check_if_it_works():
    return(list(numerical_cols))

@app.post("/predict/apply")
#application:Loan basically tells the API to expect an input in the form of the Loan class
def predict(application:Loan):

    #Model_dump is similar to application.dict()
    input_dict = application.model_dump()
    input_df = pd.DataFrame([input_dict])
    
    # Encode categorical variables (same as during training)
    encoded_input = pd.get_dummies(input_df, columns=categorical_var, drop_first=True)
    
    # Ensure all expected columns are present (add missing ones with 0)
    for col in model_features:
        if col not in encoded_input.columns:
            encoded_input[col] = 0
    
    # Reorder columns to match training data
    encoded_input = encoded_input[model_features]
    
    # Standardize numerical columns (transform only, no fitting)
    encoded_input[numerical_cols] = scaler.transform(encoded_input[numerical_cols])
    
    # Predict the risk
    risk_pred = round(regressor.predict(encoded_input)[0], 3)
    
    # Predict the approval
    approval_pred = int(LinearSVC.predict(encoded_input)[0])
    
    if (approval_pred == 0):
        approval_pred = str("Deny!")
    else:
        approval_pred = str("Approve!")
    
    return {
        "The risk score is:":(risk_pred),
        "Loan Approval Status:":approval_pred
    }

