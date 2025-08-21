# Heart Disease Prediction System

A Streamlit web application that uses machine learning to predict heart disease risk based on patient medical data.

## Features

- **Model Training**: Upload CSV data to train a logistic regression model
- **Batch Prediction**: Upload CSV files for multiple patient predictions
- **Individual Prediction**: Manual input form for single patient assessment
- **Results Export**: Download prediction results as CSV files
- **Interactive Interface**: User-friendly web interface with real-time feedback

## Installation

1. Clone or download the project files
2. Install required dependencies:
```bash
pip install streamlit pandas numpy scikit-learn
```

3. Run the application:
```bash
streamlit run heart_disease_app.py
```

## Dataset Requirements

### Training Data Format
Your training CSV file must contain the following 14 columns:

| Column | Description | Values |
|--------|-------------|---------|
| age | Age in years | Numeric (29-77) |
| sex | Gender | 0 = Female, 1 = Male |
| cp | Chest pain type | 0-3 |
| trestbps | Resting blood pressure (mm Hg) | Numeric (94-200) |
| chol | Serum cholesterol (mg/dl) | Numeric (126-564) |
| fbs | Fasting blood sugar > 120 mg/dl | 0 = No, 1 = Yes |
| restecg | Resting ECG results | 0-2 |
| thalach | Maximum heart rate achieved | Numeric (71-202) |
| exang | Exercise induced angina | 0 = No, 1 = Yes |
| oldpeak | ST depression induced by exercise | Numeric (0.0-6.2) |
| slope | Slope of peak exercise ST segment | 0-2 |
| ca | Number of major vessels (0-4) | 0-4 |
| thal | Thalassemia | 0-3 |
| target | Heart disease diagnosis | 0 = No Disease, 1 = Disease |

### Prediction Data Format
For making predictions, include all columns except `target`. If `target` is included, the app will calculate prediction accuracy.

## Usage Guide

### 1. Train Model & Predict
1. Navigate to the "Train Model & Predict" tab
2. Upload a CSV file with training data (must include `target` column)
3. Review dataset preview and statistics
4. Click "Train Model" to build the logistic regression model
5. Upload a separate CSV file for predictions (optional)
6. Download results if needed

### 2. Single Patient Prediction
1. Navigate to "Single Patient Prediction" tab
2. Ensure a model has been trained first
3. Enter patient information in the form
4. Click "Predict Heart Disease Risk"
5. View results with probability scores and recommendations

### 3. About Dataset
- View detailed information about dataset features
- Understand model specifications and limitations

## Model Performance

- **Algorithm**: Logistic Regression with L-BFGS solver
- **Typical Accuracy**: 80-85% on test data
- **Training/Test Split**: 80/20 with stratified sampling
- **Cross-validation**: Built-in train-test evaluation

## File Structure

```
heart_disease_app.py    # Main Streamlit application
README.md              # This file
requirements.txt       # Python dependencies (optional)
```

## Sample Data

The application expects data similar to the Cleveland Heart Disease dataset format. A typical row might look like:
```csv
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
63,1,3,145,233,1,0,150,0,2.3,0,0,1,1
```

## Error Handling

The application includes validation for:
- Missing or incorrect column names
- Invalid data types
- File format issues
- Model training failures

## Limitations and Disclaimers

⚠️ **Important Medical Disclaimer**: This application is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.

**Technical Limitations**:
- Model performance depends on training data quality
- Predictions are probabilistic, not definitive diagnoses
- Requires specific data format for optimal performance
- Not validated for clinical use

## Contributing

To improve the application:
1. Add more sophisticated ML algorithms (Random Forest, XGBoost)
2. Implement feature importance visualization
3. Add data preprocessing options
4. Include model comparison metrics
5. Enhance UI/UX design

## Dependencies

- `streamlit >= 1.28.0`
- `pandas >= 2.0.0`
- `numpy >= 1.24.0`
- `scikit-learn >= 1.3.0`

## License

This project is for educational purposes. Please ensure appropriate licensing for any commercial use.

## Support

For technical issues:
1. Verify all dependencies are installed correctly
2. Check CSV file format matches requirements
3. Ensure Python version compatibility (3.8+)
4. Review error messages in the Streamlit interface

## Version History

- v1.0: Initial release with basic prediction functionality
- Features: Model training, batch prediction, single patient input, results export
