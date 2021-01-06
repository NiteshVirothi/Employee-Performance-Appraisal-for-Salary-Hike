from flask import Flask, render_template,request
import requests
import pandas as pdb
import numpy as np
import joblib


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/employee_hch')
def employeeHch():
    return render_template('employee_hch.html')

@app.route('/predict_employee_hch',methods=['GET','POST'])
def employeePredictHch():
    if request.method =='POST':
       
            MarriedID = int(request.form['MarriedID'])
            MaritalStatusID = int(request.form['MaritalStatusID'])
            GenderID = int(request.form['GenderID'])
            EmpStatusID = int(request.form['EmpStatusID'])
           
           
            PayRate = int(request.form['PayRate'])
            
            PositionID = int(request.form['PositionID'])
            
            EmpSatisfaction = int(request.form['EmpSatisfaction'])
            SpecialProjectsCount = int(request.form['SpecialProjectsCount'])
            PerfScoreID = int(request.form['PerfScoreID'])
            pred_args =[MarriedID,MaritalStatusID,GenderID,EmpStatusID,PayRate,PositionID,EmpSatisfaction,SpecialProjectsCount,PerfScoreID]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1,-1)
            dec_classi = open("employeehike.pkl","rb")
            dec_model = joblib.load(dec_classi)
            model_prediction = dec_model.predict(pred_args_arr)
            model_prediction = int(model_prediction)
            return render_template("predict_employee_hch.html",prediction = model_prediction)


















@app.route('/employee_pch')
def employeePch():
    return render_template('employee_pch.html')

@app.route('/predict_employee_pch',methods=['GET','POST'])
def employeePredictPch():
    if request.method =='POST':
       
            MarriedID = int(request.form['MarriedID'])
            MaritalStatusID = int(request.form['MaritalStatusID'])
            GenderID = int(request.form['GenderID'])
            EmpStatusID = int(request.form['EmpStatusID'])
            DeptID = int(request.form['DeptID'])
            FromDiversityJobFairID = int(request.form['FromDiversityJobFairID'])
            PayRate = int(request.form['PayRate'])
            Termd = int(request.form['Termd'])
            PositionID = int(request.form['PositionID'])
            ManagerID = int(request.form['ManagerID'])
            EmpSatisfaction = int(request.form['EmpSatisfaction'])
            SpecialProjectsCount = int(request.form['SpecialProjectsCount'])
            pred_args =[MarriedID,MaritalStatusID,GenderID,EmpStatusID,DeptID,FromDiversityJobFairID,PayRate,Termd,PositionID,ManagerID,EmpSatisfaction,SpecialProjectsCount]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1,-1)
            dec_classi = open("employeeperformscore.pkl","rb")
            dec_model = joblib.load(dec_classi)
            model_prediction = dec_model.predict(pred_args_arr)
            model_prediction = int(model_prediction)
            return render_template("predict_employee_pch.html",prediction = model_prediction)

@app.route('/predict_employee_ach',methods=['GET','POST'])
def employeePredictAch():
    if request.method =='POST':
       
            Age = int(request.form['Age'])
            BusinessTravel = int(request.form['BusinessTravel'])
            DailyRate = int(request.form['DailyRate'])
            Department = int(request.form['Department'])
            DistanceFromHome = int(request.form['DistanceFromHome'])
            Education = int(request.form['Education'])
            EducationField = int(request.form['EducationField'])
            EmployeeCount = int(request.form['EmployeeCount'])
            EmployeeNumber = int(request.form['EmployeeNumber'])
            EnvironmentSatisfaction = int(request.form['EnvironmentSatisfaction'])
            Gender = int(request.form['Gender'])
            HourlyRate = int(request.form['HourlyRate'])
            JobInvolvement = int(request.form['JobInvolvement'])
            JobLevel = int(request.form['JobLevel'])
            JobRole = int(request.form['JobRole'])
            JobSatisfaction = int(request.form['JobSatisfaction'])
            MaritalStatus = int(request.form['MaritalStatus'])
            MonthlyIncome = int(request.form['MonthlyIncome'])
            MonthlyRate = int(request.form['MonthlyRate'])
            NumCompaniesWorked = int(request.form['NumCompaniesWorked'])
            Over18 = int(request.form['Over18'])
            OverTime = int(request.form['OverTime'])
            PercentSalaryHike = int(request.form['PercentSalaryHike'])
            PerformanceRating = int(request.form['PerformanceRating'])
            RelationshipSatisfaction = int(request.form['RelationshipSatisfaction'])
            StandardHours = int(request.form['StandardHours'])
            StockOptionLevel = int(request.form['StockOptionLevel'])
            TotalWorkingYears = int(request.form['TotalWorkingYears'])
            TrainingTimesLastYear = int(request.form['TrainingTimesLastYear'])
            WorkLifeBalance = int(request.form['WorkLifeBalance'])
            YearsAtCompany = int(request.form['YearsAtCompany'])
            YearsInCurrentRole = int(request.form['YearsInCurrentRole'])
            YearsSinceLastPromotion = int(request.form['YearsSinceLastPromotion'])
            YearsWithCurrManager = int(request.form['YearsWithCurrManager'])
           
            
            pred_args =[Age,BusinessTravel,DailyRate,Department,DistanceFromHome,Education,EducationField,EmployeeCount,EmployeeNumber,EnvironmentSatisfaction,Gender,HourlyRate,JobInvolvement,JobLevel,JobRole,JobSatisfaction,MaritalStatus,MonthlyIncome,MonthlyRate,NumCompaniesWorked,Over18,OverTime,PercentSalaryHike,PerformanceRating,RelationshipSatisfaction,StandardHours,StockOptionLevel,TotalWorkingYears,TrainingTimesLastYear,WorkLifeBalance,YearsAtCompany,YearsInCurrentRole,YearsSinceLastPromotion,YearsWithCurrManager]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1,-1)
            dec_classi = open("employeeattrmodel.pkl","rb")
            dec_model = joblib.load(dec_classi)
            model_prediction = dec_model.predict(pred_args_arr)
            model_prediction = int(model_prediction)
            return render_template("predict_employee_ach.html",prediction = model_prediction)

@app.route('/predict_employee_sch',methods=['GET','POST'])
def employeePredictSch():
    if request.method =='POST':
       
            Company = int(request.form['company'])
            Job = int(request.form['job'])
            Degree = int(request.form['degree'])
         
            pred_args =[Company,Job,Degree]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1,-1)
            dec_classi = open("employeesalaryprediction.pkl","rb")
            dec_model = joblib.load(dec_classi)
            model_prediction = dec_model.predict(pred_args_arr)
            model_prediction = int(model_prediction)
            return render_template("predict_employee_sch.html",prediction = model_prediction)


@app.route('/employee_ach')
def employeeAch():
    return render_template('employee_ach.html')


@app.route('/employee_sch')
def employeeSch():
    return render_template('employee_sch.html')

@app.route('/employee_vch')
def employeeVch():
    return render_template('employee_vch.html')


@app.route('/employee_gch')
def employeeGch():
    return render_template('employee_gch.html')

if __name__ == "__main__":
    app.run()