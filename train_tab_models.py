import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def train_diabetes():
    print("--- Training Diabetes Model ---")
    df = pd.read_csv("diabetes_prediction_dataset.csv")
    
    # Mappings based on app.py components
    # gender: Female=0, Male=1 (Other=2?)
    le_gender = LabelEncoder()
    df['gender'] = le_gender.fit_transform(df['gender'])
    # Force specific mapping if possible? 
    # Let's trust LabelEncoder but stick to common sense: usually F=0, M=1 if sorted?
    # Female, Male -> F comes before M -> 0=Female, 1=Male. Correct.
    
    # smoking_history: various values
    le_smoking = LabelEncoder()
    df['smoking_history'] = le_smoking.fit_transform(df['smoking_history'])
    print("Diabetes Smoking Mapping:", dict(zip(le_smoking.classes_, le_smoking.transform(le_smoking.classes_))))
    
    # Features as per app.py order:
    # gender, age, hypertension, heart_disease_history, smoking_history, bmi, HbA1c_level, blood_glucose
    # CSV cols: gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level
    
    X = df[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
    y = df['diabetes']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    print(f"Diabetes Accuracy: {accuracy_score(y_test, model.predict(X_test_scaled))}")
    
    with open("diabetes_gb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("diabetes_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Diabetes model saved.\n")


def train_heart():
    print("--- Training Heart Model ---")
    df = pd.read_csv("heart_disease_dataset.csv")
    
    # CSV Headers: Age,Gender,Cholesterol,Blood Pressure,Heart Rate,Smoking,Alcohol Intake,
    # Exercise Hours,Family History,Diabetes,Obesity,Stress Level,Blood Sugar,
    # Exercise Induced Angina,Chest Pain Type,Heart Disease
    
    # app.py Features (14):
    # age, gender, cholesterol, blood_pressure, heart_rate, smoking_history, alcohol_intake, 
    # exercise_hours, diabetes, obesity, stress_level, blood_sugar, exercise, family_history
    
    # Mapping CSV to app features:
    # Gender: Male/Female -> 1/0?
    # app.py says: 0=Female, 1=Male.
    # LabelEncoder: Female=0, Male=1. Matches.
    df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
    
    # Smoking: Current/Never/Former...
    df['Smoking'] = LabelEncoder().fit_transform(df['Smoking'])
    
    # Alcohol Intake: ? Check unique values or assume encoding needed or numeric? 
    # Usually Strings like 'Heavy', 'None'.
    if df['Alcohol Intake'].dtype == object:
        df['Alcohol Intake'] = LabelEncoder().fit_transform(df['Alcohol Intake'])
        
    # Diabetes, Obesity, Family History, Exercise Induced Angina: 'Yes'/'No' -> 1/0
    binary_cols = ['Diabetes', 'Obesity', 'Family History', 'Exercise Induced Angina']
    for col in binary_cols:
        if df[col].dtype == object:
             # Yes=1, No=0.
             df[col] = df[col].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    
    # Features selection
    # Note: 'Exercise Induced Angina' maps to 'exercise' in app.py context likely
    X = df[['Age', 'Gender', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Smoking', 'Alcohol Intake', 
            'Exercise Hours', 'Diabetes', 'Obesity', 'Stress Level', 'Blood Sugar', 'Exercise Induced Angina', 'Family History']]
            
    y = df['Heart Disease']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) # Use full data for robustness or split? Let's split.
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    print(f"Heart Accuracy: {accuracy_score(y_test, model.predict(X_test))}")
    
    with open("heart_nb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("heart_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Heart model saved.\n")


def train_stroke():
    print("--- Training Stroke Model ---")
    df = pd.read_csv("stroke_prediction_dataset.csv")
    
    # Drop ID
    df = df.drop(columns=['id'])
    
    # Handle BMI N/A
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    df['bmi'].fillna(df['bmi'].mean(), inplace=True)
    
    # Encodings
    # gender: Male=1, Female=0
    df = df[df['gender'] != 'Other'] # Drop 'Other' which is rare and messes up binary logic
    df['gender'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)
    
    # ever_married: Yes=1, No=0
    df['ever_married'] = df['ever_married'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Residence_type: Urban=1, Rural=0 (app.py comments: 0=Rural, 1=Urban)
    df['Residence_type'] = df['Residence_type'].apply(lambda x: 1 if x == 'Urban' else 0)
    
    # work_type: Categorical
    le_work = LabelEncoder()
    df['work_type'] = le_work.fit_transform(df['work_type'])
    print("Stroke Work Type Mapping:", dict(zip(le_work.classes_, le_work.transform(le_work.classes_))))
    
    # smoking_status: Categorical
    le_smoke = LabelEncoder()
    df['smoking_status'] = le_smoke.fit_transform(df['smoking_status'])
    print("Stroke Smoking Mapping:", dict(zip(le_smoke.classes_, le_smoke.transform(le_smoke.classes_))))

    # app.py features:
    # gender, age, hypertension, heart_disease, ever_married, work_type, residence, avg_glucose, bmi, smoking_history
    X = df[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']]
    y = df['stroke']
    
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    print(f"Stroke Accuracy: {accuracy_score(y_test, model.predict(X_test_scaled))}")
    
    with open("stroke_gb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("stroke_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Stroke model saved.\n")

if __name__ == "__main__":
    train_diabetes()
    train_heart()
    train_stroke()
    print("All models retrained successfully.")
