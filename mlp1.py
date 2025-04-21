import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_diabetes, load_breast_cancer
from streamlit_option_menu import option_menu
import numpy as np

# Streamlit UI
with st.sidebar:
    selected = option_menu("Select a Model to Predict", 
                           ["Home", "Predict Heart Disease", "Predict Stroke", "Predict Breast Cancer", "Predict Diabetes", "Predict Kidney Disease", "Predict Other Diseases"], 
                           icons=['home', 'heart', 'activity', 'gender-female', 'droplet', 'pen', 'paper'], 
                           menu_icon="cast", 
                           default_index=0)

# Load the datasets
heart_file_path = 'preprocessed_heart_disease_data_set.csv'
stroke_file_path = 'preprocessed_stroke_prediction_data.csv'
kidney_file_path = 'preprocessed_data_set_kidney.csv'


heart_dataset = pd.read_csv(heart_file_path).drop(columns=['Unnamed: 0'], errors='ignore')
heart_dataset = heart_dataset[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope', 'HeartDisease']]
heart_data_columns = heart_dataset.columns

stroke_dataset = pd.read_csv(stroke_file_path).drop(columns=['Unnamed: 0'], errors='ignore')

# Load breast cancer dataset
breast_cancer = load_breast_cancer()
breast_features = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
breast_target = pd.Series(breast_cancer.target)

# Load diabetes dataset
diabetes = load_diabetes()
diabetes_features = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
diabetes_target = pd.Series(diabetes.target)

# Load kidney dataset
kidney_dataset = pd.read_csv(kidney_file_path).drop(columns=['Unnamed: 0'], errors='ignore')
kidney_features = kidney_dataset.drop('class', axis=1)
kidney_target = kidney_dataset['class']

# Train heart disease prediction model
heart_features = heart_dataset[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']]
heart_target = heart_dataset['HeartDisease']
heart_train_data, heart_test_data, heart_train_target, heart_test_target = train_test_split(heart_features, heart_target, test_size=0.2)
heart_rf = RandomForestClassifier(n_estimators=100)
heart_rf.fit(heart_train_data, heart_train_target)

# Train stroke prediction model
stroke_features = stroke_dataset[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'avg_glucose_level', 'bmi', 'smoking_status']]
stroke_target = stroke_dataset['stroke']
stroke_train_data, stroke_test_data, stroke_train_target, stroke_test_target = train_test_split(stroke_features, stroke_target, test_size=0.2)
stroke_rf = RandomForestClassifier(n_estimators=100)
stroke_rf.fit(stroke_train_data, stroke_train_target)

# Train breast cancer prediction model
breast_train_data, breast_test_data, breast_train_target, breast_test_target = train_test_split(breast_features, breast_target, test_size=0.2)
breast_rf = RandomForestClassifier(n_estimators=100)
breast_rf.fit(breast_train_data, breast_train_target)

# Train diabetes prediction model
diabetes_train_data, diabetes_test_data, diabetes_train_target, diabetes_test_target = train_test_split(diabetes_features, diabetes_target, test_size=0.2)
diabetes_rf = RandomForestClassifier(n_estimators=100)
diabetes_rf.fit(diabetes_train_data, diabetes_train_target)

# Train kidney disease prediction model
kidney_train_data, kidney_test_data, kidney_train_target, kidney_test_target = train_test_split(kidney_features, kidney_target, test_size=0.2)
kidney_rf = RandomForestClassifier(n_estimators=100)
kidney_rf.fit(kidney_train_data, kidney_train_target)

mapping_dict = {
    'bp (Diastolic)': {'0': 0, '1': 1},
    'bp limit': {'0': 0, '1': 1, '2': 2},
    'sg': {'< 1.007': 0, '1.009 - 1.011': 1, '1.015 - 1.017': 2, '1.019 - 1.021': 3, '≥ 1.023': 4},
    'al': {'< 0': 0, '1 - 1': 1, '2 - 2': 2, '3 - 3': 3, '≥ 4': 4},
    'class': {'ckd': 1, 'notckd': 0},
    'rbc': {'0': 0, '1': 1},
    'su': {'< 0': 0, '1 - 2': 1, '2 - 2': 2, '3 - 4': 3, '4 - 4': 4, '≥ 4': 5},
    'pc': {'0': 0, '1': 1},
    'pcc': {'0': 0, '1': 1},
    'ba': {'0': 0, '1': 1},
    'bgr': {'< 112': 0, '112 - 154': 1, '154 - 196': 2, '196 - 238': 3, '238 - 280': 4, '280 - 322': 5, '322 - 364': 6, '364 - 406': 7, '406 - 448': 8, '≥ 448': 9},
    'bu': {'< 48.1': 0, '48.1 - 86.2': 1, '86.2 - 124.3': 2, '124.3 - 162.4': 3, '162.4 - 200.5': 4, '200.5 - 238.6': 5, '238.6 - 276.7': 6, '≥ 352.9': 7},
    'sod': {'< 118': 0, '118 - 123': 1, '123 - 128': 2, '128 - 133': 3, '133 - 138': 4, '138 - 143': 5, '143 - 148': 6, '148 - 153': 7, '≥ 158': 8},
    'sc': {'< 3.65': 0, '3.65 - 6.8': 1, '6.8 - 9.95': 2, '9.95 - 13.1': 3, '13.1 - 16.25': 4, '16.25 - 19.4': 5, '≥ 28.85': 6},
    'pot': {'< 7.31': 0, '7.31 - 11.72': 1, '11.72 - 16.13': 2, '16.13 - 20.54': 3, '20.54 - 24.95': 4, '24.95 - 29.36': 5, '29.36 - 33.77': 6, '33.77 - 38.18': 7, '38.18 - 42.59': 8, '≥ 42.59': 9},
    'hemo': {'< 6.1': 0, '6.1 - 7.4': 1, '7.4 - 8.7': 2, '8.7 - 10': 3, '10 - 11.3': 4, '11.3 - 12.6': 5, '12.6 - 13.9': 6, '13.9 - 15.2': 7, '15.2 - 16.5': 8, '≥ 16.5': 9},
    'pcv': {'< 17.9': 0, '17.9 - 21.8': 1, '21.8 - 25.7': 2, '25.7 - 29.6': 3, '29.6 - 33.5': 4, '33.5 - 37.4': 5, '37.4 - 41.3': 6, '41.3 - 45.2': 7, '45.2 - 49.1': 8, '≥ 49.1': 9},
    'rbcc': {'< 2.69': 0, '2.69 - 3.28': 1, '3.28 - 3.87': 2, '3.87 - 4.46': 3, '4.46 - 5.05': 4, '5.05 - 5.64': 5, '5.64 - 6.23': 6, '≥ 6.23': 7},
    'wbcc': {'< 6200': 0, '6200 - 9400': 1, '9400 - 12600': 2, '12600 - 15800': 3, '15800 - 19000': 4, '19000 - 22200': 5, '22200 - 25400': 6, '25400 - 28600': 7, '28600 - 31800': 8, '≥ 31800': 9},
    'htn': {'0': 0, '1': 1},
    'dm': {'0': 0, '1': 1},
    'cad': {'0': 0, '1': 1},
    'appet': {'0': 0, '1': 1},
    'pe': {'0': 0, '1': 1},
    'ane': {'0': 0, '1': 1},
    'grf': {'< 26.6175': 0, '26.6175 - 51.7832': 1, '51.7832 - 76.949': 2, '76.949 - 102.115': 3, '102.115 - 127.281': 4, '127.281 - 152.446': 5, '152.446 - 177.612': 6, '177.612 - 202.778': 7, '202.778 - 227.944': 8, '≥ 227.944': 9},
    'stage': {'s1': 1, 's2': 2, 's3': 3, 's4': 4, 's5': 5},
    'affected': {'class': -1, '0': 0, '1': 1},
    'age': {'< 12': 0, '12 - 20': 1, '20 - 27': 2, '27 - 35': 3, '35 - 43': 4, '43 - 51': 5, '51 - 59': 6, '59 - 66': 7, '66 - 74': 8, '≥ 74': 9}
}
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
def jaccard_similarity(selected_symptoms, disease_symptoms):
    selected_symptoms_set = set(selected_symptoms)
    disease_symptoms_set = set(disease_symptoms)
    intersection = len(selected_symptoms_set.intersection(disease_symptoms_set))
    union = len(selected_symptoms_set.union(disease_symptoms_set))
    return intersection / union if union > 0 else 0

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
disease_prediction=pd.read_csv("disease_symptoms_preprocessed.csv")
diseases_list=list(disease_prediction['Disease'].unique())
disease_symptoms_dict={}
disease_prediction_list=[]
for kalki in diseases_list:
    duplicate_list=[list(x) for x in disease_prediction[disease_prediction['Disease']==kalki].to_numpy()]
    unique_list=[]
    for i in duplicate_list:
        del i[0]
        for j in i:
            if j not in unique_list and j != np.nan:
                unique_list.append(j)
    disease_symptoms_dict[kalki]=unique_list
all_symptoms=[]
for i in disease_symptoms_dict.values():
    for j in i:
        if j not in all_symptoms:
            all_symptoms.append(j)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

if selected == "Predict Heart Disease":
    st.title("Heart Disease Prediction")
    tab1, tab2 = st.tabs(["Enter Data", "Check Your Result"])
    with tab1:
        age = st.number_input("Enter Age", min_value=0, max_value=100, value=50, step=1)
        sex = st.selectbox("Select Sex", ["Male", "Female"])
        cp = st.selectbox("Select Chest Pain Type", heart_dataset['ChestPainType'].unique())
        trestbps = st.number_input("Enter Resting Blood Pressure", min_value=0, max_value=200, value=120, step=1)
        chol = st.number_input("Enter Cholesterol", min_value=0, max_value=600, value=200, step=1)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
        restecg = st.selectbox("Resting Electrocardiographic Results", heart_dataset['RestingECG'].unique())
        thalach = st.number_input("Enter Maximum Heart Rate Achieved", min_value=0, max_value=220, value=150, step=1)
        exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
        oldpeak = st.number_input("Enter ST depression induced by exercise relative to rest", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Select the slope of the peak exercise ST segment", heart_dataset['ST_Slope'].unique())
        proceed = st.checkbox("Predict whether I have heart disease")
        if proceed:
            with tab2:
                sex = 1 if sex == "Male" else 0
                fbs = 1 if fbs == "Yes" else 0
                exang = 1 if exang == "Yes" else 0
                values_list = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]
                input_data = pd.DataFrame([values_list], columns=heart_features.columns)
                predicted_value = heart_rf.predict(input_data)
                st.write("Prediction (Heart Disease: 1, No Heart Disease: 0):", predicted_value[0])
                col1,col2=st.columns([1,1])
                col3,col4=st.columns([1,1])
                st.subheader("Symptoms & treatment, youtube videos")
                st.success("These videos are for inormation purposes only before trying these, please consult doctor")
                with col1:
                    st.video("https://youtu.be/J1DUQFL-VHw?si=q-BnzqKXG8HTdDX8")
                with col2:
                    st.video("https://youtu.be/LPoOFqYr6vE?si=Qd5937X3W6VR0u20")
                with col3:
                    st.video("https://youtu.be/y7hiVD53aBU?si=4uxV62Ouban1Ivgn")
                with col4:
                    st.video("https://youtu.be/aXDaBuPSvJs?si=HpKyLoX3O3eHeY5C")

elif selected == "Predict Stroke":
    st.title("Stroke Prediction")
    tab1, tab2 = st.tabs(["Enter Data", "Check Your Result"])
    with tab1:
        gender = st.selectbox("Select Gender", stroke_dataset['gender'].unique())
        age = st.number_input("Enter Age", min_value=0, max_value=100, value=50, step=1)
        hypertension = st.selectbox("Hypertension", ["Yes", "No"])
        heart_disease = st.selectbox("Heart Disease", ["Yes", "No"])
        ever_married = st.selectbox("Ever Married", ["Yes", "No"])
        avg_glucose_level = st.number_input("Enter Average Glucose Level", min_value=0.0, max_value=300.0, value=100.0, step=0.1)
        bmi = st.number_input("Enter Body Mass Index (BMI)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
        smoking_status = st.selectbox("Select Smoking Status", stroke_dataset['smoking_status'].unique())
        proceed = st.checkbox("Predict whether I have stroke")
        if proceed:
            with tab2:
                hypertension = 1 if hypertension == "Yes" else 0
                heart_disease = 1 if heart_disease == "Yes" else 0
                ever_married = 1 if ever_married == "Yes" else 0
                values_list = [gender, age, hypertension, heart_disease, ever_married, avg_glucose_level, bmi, smoking_status]
                input_data = pd.DataFrame([values_list], columns=stroke_features.columns)
                predicted_value = stroke_rf.predict(input_data)
                st.write("Prediction (Stroke: 1, No Stroke: 0):", predicted_value[0])
                st.divider()
                if predicted_value[0]==1:
                    st.warning("Sorry, to say this you might be suffering from heart realted issues,please consult the doctor for better health care services")
                else:
                    st.success("Hey, you don't have any stroke related problems")
                col1,col2=st.columns([1,1])
                col3,col4=st.columns([1,1])
                st.subheader("Symptoms & treatment, youtube videos")
                st.success("These videos are for inormation purposes only before trying these, please consult doctor")
                with col1:
                    st.video("https://youtu.be/mkpbbWZvYmw?si=w0-5I-rccH2P7lIb")
                with col2:
                    st.video("https://youtu.be/mBPJp7H4bo8?si=IMO9UXc7p36ogGjh")
                with col3:
                    st.video("https://youtu.be/EY98RInP-A4?si=HP7xIyn9BIKZGvzW")
                with col4:
                    st.video("https://youtu.be/CP-WEh9C9Eg?si=xMwgb92sl9Y0cO9_")

elif selected == "Predict Breast Cancer":
    st.title("Breast Cancer Prediction")
    tab1, tab2 = st.tabs(["Enter Data", "Check Your Result"])
    with tab1:
        inputs = {}
        for feature in breast_features.columns:
            inputs[feature] = st.number_input(f"Enter {feature.replace('_', ' ')}", key=f"breast_{feature}")
        proceed = st.checkbox("Predict whether I have breast cancer", key="breast_cancer_prediction")
        if proceed:
            with tab2:
                values_list = [inputs[feature] for feature in breast_features.columns]
                input_data = pd.DataFrame([values_list], columns=breast_features.columns)
                predicted_value = breast_rf.predict(input_data)
                st.write("Prediction (Breast Cancer: 1, No Breast Cancer: 0):", predicted_value[0])
                st.divider()
                if predicted_value[0]==1:
                    st.warning("Sorry, to say this you might be suffering from breast cancer issues,please consult the doctor for better health care services")
                else:
                    st.success("Hey, you don't have any breast cancer realted problems")
                col1,col2=st.columns([1,1])
                col3,col4=st.columns([1,1])
                st.subheader("Symptoms & treatment, youtube videos")
                st.success("These videos are for inormation purposes only before trying these, please consult doctor")
                with col1:
                    st.video("https://youtu.be/XhZQfMMCn80?si=GPjfxbBNDMt71bFr")
                with col2:
                    st.video("https://youtu.be/8T90GKkoqZ4?si=t27Ccydc1vrs9qkm")
                with col3:
                    st.video("https://youtu.be/yyY5A7HnGrA?si=a1xNr2oUS4S1j2bK")
                with col4:
                    st.video("https://youtu.be/9E_-dLYl6nE?si=r1fgySvF6G2opzao")


elif selected == "Predict Diabetes":
    st.title("Diabetes Prediction")
    tab1, tab2 = st.tabs(["Enter Data", "Check Your Result"])
    with tab1:
        inputs = {}
        for feature in diabetes_features.columns:
            inputs[feature] = st.number_input(f"Enter {feature.replace('_', ' ')}", key=f"diabetes_{feature}")
        proceed = st.checkbox("Predict whether I have diabetes", key="diabetes_prediction")
        if proceed:
            with tab2:
                values_list = [inputs[feature] for feature in diabetes_features.columns]
                input_data = pd.DataFrame([values_list], columns=diabetes_features.columns)
                predicted_value = diabetes_rf.predict(input_data)
                st.write("Prediction (Diabetes: 1, No Diabetes: 0):", predicted_value[0])
                st.divider()
                if predicted_value[0]==1:
                    st.warning("Sorry, to say this you might be suffering from diabetic issues,please consult the doctor for better health care services")
                else:
                    st.success("Hey, you don't have any diabetic realted problems")
                col1,col2=st.columns([1,1])
                col3,col4=st.columns([1,1])
                st.subheader("Symptoms & treatment, youtube videos")
                st.success("These videos are for inormation purposes only before trying these, please consult doctor")
                with col1:
                    st.video("https://youtu.be/bblQecVzvxY?si=6ZAKketBm1zDajmU")
                with col2:
                    st.video("https://youtu.be/503WXGa7uxc?si=8RKx-6lCxC7bqTNQ")
                with col3:
                    st.video("https://youtu.be/botZnpsTfZ0?si=jTVbyq_vSBe-LBwH")
                with col4:
                    st.video("https://youtu.be/w4zSBSM0Vhk?si=1SnK14oNDUlaCDUj")


elif selected == "Predict Kidney Disease":
    st.title("Kidney Disease Prediction")
    tab1, tab2 = st.tabs(["Enter Data", "Check Your Result"])
    with tab1:
        inputs = {}
        for feature in kidney_features.columns:
            options = list(mapping_dict[feature].keys())
            selected_option = st.selectbox(f"Select {feature.replace('_', ' ')}", options, key=f"kidney_{feature}")
            inputs[feature] = mapping_dict[feature][selected_option]
        proceed = st.checkbox("Predict whether I have kidney disease", key="kidney_disease_prediction")
        if proceed:
            with tab2:
                values_list = [inputs[feature] for feature in kidney_features.columns]
                input_data = pd.DataFrame([values_list], columns=kidney_features.columns)
                predicted_value = kidney_rf.predict(input_data)
                st.write("Prediction (Kidney Disease: 1, No Kidney Disease: 0):", predicted_value[0])
                st.divider()
                if predicted_value[0]==1:
                    st.warning("Sorry, to say this you might be suffering from diabetic issues,please consult the doctor for better health care services")
                else:
                    st.success("Hey, you don't have any diabetic realted problems")
                col1,col2=st.columns([1,1])
                col3,col4=st.columns([1,1])
                st.subheader("Symptoms & treatment, youtube videos")
                st.success("These videos are for inormation purposes only before trying these, please consult doctor")
                with col1:
                    st.video("https://youtu.be/fv53QZRk4hs?si=_2Zsb9llPuHK2IMe")
                with col2:
                    st.video("https://youtu.be/eVP6BFUd91s?si=rUgyjlJpGCxYehNU")
                with col3:
                    st.video("https://youtu.be/fUrQkLPDFzo?si=Wl45daY1e2SVnmFg")
                with col4:
                    st.video("https://youtu.be/_E1FXNGswoU?si=aOK3ccND-HnwPKz2")

elif selected == "Predict Other Diseases":
    st.title("Select all the symptoms you have")
    tab1, tab2 = st.tabs(["Select Symptoms", "Predicted Disease"])

    with tab1:
        selected_symptoms = st.multiselect("Select all the symptoms you have", all_symptoms, key="multiselect_symptoms")

    with tab2:
        disease_similarity_dict = {}
        for disease, symptoms in disease_symptoms_dict.items():
            similarity = jaccard_similarity(selected_symptoms, symptoms)
            disease_similarity_dict[disease] = similarity

        if disease_similarity_dict:
            predicted_disease = max(disease_similarity_dict, key=disease_similarity_dict.get)
            st.warning(f"We predicted that you are having {predicted_disease} with similarity score {disease_similarity_dict[predicted_disease]:.2f}")
            st.write(f"In our data base for the disease :{predicted_disease}\n we have the following above symptoms list\n {st.table(disease_symptoms_dict[predicted_disease])}")
        else:
            st.warning("No disease matches the selected symptoms.")
elif selected == "Home":
    st.title("Welcome to the Disease Prediction App!")
    st.write("""
    This application uses machine learning models to predict the likelihood of various diseases based on user-provided input data. 
    You can use the sidebar to navigate to different disease prediction models.

    ### About the Developer
    My name is Rohith Palyam. I am passionate about artificial intelligence, machine learning and its applications in healthcare. This project is an effort to provide a useful tool for preliminary health assessments. 

    ### Disclaimer
    This app is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any medical concerns.
    Because, the datasets which i used in this project are not so standered but they are really good in predicting the diseases. the datasets i used for this proect are displayed below. Hence my advice is dont rely purely on this web site even even though this website is good enough to predict the disease.This is for prileminery health checkup purpose only. It is good to consult the doctor if the situation is severe.
    The datsets I used for this project are downloaded from kaggle.
    Actually i done this project as a part of my internship on Artiicial Intelligence and Machine learning From Innovate which is approved by AICTE.
    """)
    st.divider()
    st.markdown("### Dataset used for kidney disease detection - Downloaded from kaggle")
    st.dataframe(pd.read_csv("ckd-dataset-v2.csv"))
    st.markdown("### Dataset used for heart disease detection - Downloaded from kaggle")
    st.dataframe(pd.read_csv("heart.csv"))
    st.markdown("### Dataset used for stroke prediction - Downloaded from kaggle")
    st.dataframe(pd.read_csv("healthcare-dataset-stroke-data.csv"))
    st.markdown("### Dataset used for breast cancer prediction - loaded from scikit learn")
    st.dataframe(breast_features)
    st.markdown("### Dataset used for diabetics prediction - loaded from scikit learn")
    st.dataframe(diabetes_features)
    st.markdown("### Dataset used for predicting diseases using symptoms - Downloaded from kaggle")
    st.dataframe(pd.read_csv("DiseaseAndSymptoms.csv"))
    st.markdown("### Preprocessed Dataset used for kidney disease detection - Preprocessed One")
    st.dataframe(pd.read_csv("preprocessed_data_set_kidney.csv"))
    st.markdown("### Preprocessed Dataset used for heart disease detection - Preprocessed One")
    st.dataframe(pd.read_csv("preprocessed_heart_disease_data_set.csv"))
    st.markdown("### Preprocessed Dataset used for stroke prediction - Preprocessed One")
    st.dataframe(pd.read_csv("preprocessed_stroke_prediction_data.csv"))
    st.markdown("### Preprocessed Dataset used for stroke prediction - Preprocessed One(used normal coding to detect)")
    st.dataframe(pd.read_csv("disease_symptoms_preprocessed.csv"))
