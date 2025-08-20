import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import io

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Title and description
st.title("❤️ Heart Disease Prediction System")
st.markdown("Upload a CSV file with patient data to predict heart disease risk")

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "Choose an option:",
    ("Train Model & Predict", "Single Patient Prediction", "About Dataset")
)

def load_and_train_model(data):
    """Train the logistic regression model on the provided data"""
    # Prepare features and target
    X = data.drop(columns='target', axis=1)
    y = data['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=2
    )
    
    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Calculate accuracies
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_accuracy = accuracy_score(train_pred, y_train)
    test_accuracy = accuracy_score(test_pred, y_test)
    
    return model, train_accuracy, test_accuracy, X_test, y_test

def predict_heart_disease(model, input_data):
    """Make prediction for input data"""
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    probability = model.predict_proba(input_array)
    return prediction[0], probability[0]

if option == "Train Model & Predict":
    st.header("Upload CSV File for Training and Prediction")
    
    # Expected columns information
    st.subheader("Expected CSV Format")
    expected_columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", 
        "restecg", "thalach", "exang", "oldpeak", 
        "slope", "ca", "thal", "target"
    ]
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Required Columns:**")
        for i, col in enumerate(expected_columns):
            if i < len(expected_columns)//2:
                st.write(f"• {col}")
    with col2:
        st.write("**Column Descriptions:**")
        descriptions = {
            "age": "Age in years",
            "sex": "Gender (1=male, 0=female)",
            "cp": "Chest pain type (0-3)",
            "trestbps": "Resting blood pressure",
            "chol": "Cholesterol level",
            "fbs": "Fasting blood sugar > 120 mg/dl",
            "target": "Heart disease (1=yes, 0=no)"
        }
        for col, desc in descriptions.items():
            st.write(f"• **{col}**: {desc}")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            st.subheader("Dataset Preview")
            st.write(f"Dataset shape: {df.shape}")
            st.dataframe(df.head())
            
            # Validate columns
            missing_cols = [col for col in expected_columns if col not in df.columns]
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
                st.stop()
            
            # Dataset statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Patients", len(df))
            with col2:
                healthy = len(df[df['target'] == 0])
                st.metric("Healthy Patients", healthy)
            with col3:
                diseased = len(df[df['target'] == 1])
                st.metric("Heart Disease Cases", diseased)
            
            # Train model button
            if st.button("Train Model", type="primary"):
                with st.spinner("Training model..."):
                    model, train_acc, test_acc, X_test, y_test = load_and_train_model(df)
                    
                    # Store model in session state
                    st.session_state.model = model
                    st.session_state.feature_names = df.drop(columns='target').columns.tolist()
                
                # Display results
                st.success("Model trained successfully!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Training Accuracy", f"{train_acc:.3f}")
                with col2:
                    st.metric("Test Accuracy", f"{test_acc:.3f}")
            
            # Prediction section
            if 'model' in st.session_state:
                st.subheader("Make Predictions")
                
                # Option 1: Upload new CSV for predictions
                st.write("**Option 1: Upload CSV file for batch predictions**")
                pred_file = st.file_uploader("Choose CSV file for predictions", type="csv", key="pred_file")
                
                if pred_file is not None:
                    pred_df = pd.read_csv(pred_file)
                    
                    # Check if target column exists (remove it if it does)
                    if 'target' in pred_df.columns:
                        pred_features = pred_df.drop(columns='target')
                        actual_labels = pred_df['target']
                        show_accuracy = True
                    else:
                        pred_features = pred_df
                        show_accuracy = False
                    
                    if st.button("Predict", key="batch_predict"):
                        predictions = st.session_state.model.predict(pred_features)
                        probabilities = st.session_state.model.predict_proba(pred_features)
                        
                        # Create results dataframe
                        results_df = pred_df.copy()
                        results_df['Prediction'] = predictions
                        results_df['Risk_Probability'] = probabilities[:, 1]
                        results_df['Prediction_Label'] = results_df['Prediction'].map({
                            0: 'No Heart Disease', 
                            1: 'Heart Disease'
                        })
                        
                        st.subheader("Prediction Results")
                        st.dataframe(results_df)
                        
                        # Show accuracy if actual labels available
                        if show_accuracy:
                            batch_accuracy = accuracy_score(actual_labels, predictions)
                            st.metric("Prediction Accuracy", f"{batch_accuracy:.3f}")
                        
                        # Download results
                        csv_buffer = io.StringIO()
                        results_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="Download Results",
                            data=csv_buffer.getvalue(),
                            file_name="heart_disease_predictions.csv",
                            mime="text/csv"
                        )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

elif option == "Single Patient Prediction":
    st.header("Single Patient Heart Disease Prediction")
    
    if 'model' not in st.session_state:
        st.warning("Please train a model first using the 'Train Model & Predict' option.")
    else:
        st.write("Enter patient information:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
            trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=250, value=120)
            chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            restecg = st.selectbox("Resting ECG", [0, 1, 2])
        
        with col2:
            thalach = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
            exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2])
            ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4])
            thal = st.selectbox("Thalassemia", [0, 1, 2, 3])
        
        if st.button("Predict Heart Disease Risk", type="primary"):
            input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            prediction, probability = predict_heart_disease(st.session_state.model, input_data)
            
            # Display results
            if prediction == 0:
                st.success("🟢 Low Risk: The model predicts NO heart disease")
                risk_level = "Low Risk"
                color = "green"
            else:
                st.error("🔴 High Risk: The model predicts HEART DISEASE")
                risk_level = "High Risk"
                color = "red"
            
            # Show probability
            risk_prob = probability[1] * 100
            st.write(f"**Risk Probability: {risk_prob:.1f}%**")
            
            # Progress bar for risk level
            st.progress(risk_prob / 100)
            
            # Recommendation
            if prediction == 1:
                st.write("**⚠️ Recommendation:** Please consult with a healthcare professional immediately.")
            else:
                st.write("**✅ Recommendation:** Maintain a healthy lifestyle and regular check-ups.")

else:  # About Dataset
    st.header("About the Heart Disease Dataset")
    
    st.write("""
    This application uses a heart disease dataset to predict the likelihood of heart disease in patients.
    The dataset contains 14 attributes that are commonly used in cardiovascular risk assessment.
    """)
    
    st.subheader("Feature Descriptions")
    
    features_info = {
        "age": "Age of the patient in years",
        "sex": "Gender (1 = male; 0 = female)",
        "cp": "Chest pain type (0-3)",
        "trestbps": "Resting blood pressure (mm Hg)",
        "chol": "Serum cholesterol (mg/dl)",
        "fbs": "Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)",
        "restecg": "Resting electrocardiographic results (0-2)",
        "thalach": "Maximum heart rate achieved",
        "exang": "Exercise induced angina (1 = yes; 0 = no)",
        "oldpeak": "ST depression induced by exercise relative to rest",
        "slope": "Slope of the peak exercise ST segment (0-2)",
        "ca": "Number of major vessels colored by fluoroscopy (0-4)",
        "thal": "Thalassemia (0-3)",
        "target": "Heart disease diagnosis (1 = disease; 0 = no disease)"
    }
    
    for feature, description in features_info.items():
        st.write(f"**{feature}**: {description}")
    
    st.subheader("Model Information")
    st.write("""
    - **Algorithm**: Logistic Regression
    - **Purpose**: Binary classification (Heart Disease vs. No Heart Disease)
    - **Performance**: Typically achieves 80-85% accuracy
    - **Use Case**: Medical screening and risk assessment
    """)
    
    st.warning("""
    **Important Disclaimer**: This tool is for educational and research purposes only. 
    It should not be used as a substitute for professional medical advice, diagnosis, or treatment. 
    Always consult with qualified healthcare professionals for medical decisions.
    """)