import pandas as pd
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

# Load the cleaned dataset
df_copy = pd.read_csv('cleaned_domestic_violence.csv')

# Load your encoded dataset (make sure to keep the feature names)
df_encoded = pd.get_dummies(df_copy, columns=['education', 'employment', 'marital_status', 'violence'], drop_first=True)

# Get the list of feature names from the encoded DataFrame (excluding the target variable)
feature_names = df_encoded.columns.tolist()
feature_names.remove('violence_yes')  # Remove the target variable

# Split data into features and target
X = df_encoded[feature_names]
y = df_encoded['violence_yes']

# Split into train and test datasets (if needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with an imputer and the gradient boosting classifier
gradient_boosting_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('model', GradientBoostingClassifier())
])

# Fit the pipeline with the training data
gradient_boosting_pipeline.fit(X_train, y_train)

# Save the fitted pipeline (optional, if you want to avoid retraining each time)
joblib.dump(gradient_boosting_pipeline, 'gradient_boosting_pipeline.pkl')

def encode_user_input(user_data):
    # Create a DataFrame for user input with all columns initialized to 0
    user_data_encoded = pd.DataFrame(columns=feature_names)

    # Add numeric user data
    user_data_encoded.loc[0, 'age'] = int(user_data['age'])  # Ensure age is int
    user_data_encoded.loc[0, 'income'] = float(user_data['income'])  # Ensure income is float

    # One-hot encode the categorical features from user input
    education_dummies = pd.get_dummies([user_data['education']], prefix='education', drop_first=True)
    employment_dummies = pd.get_dummies([user_data['employment']], prefix='employment', drop_first=True)
    marital_status_dummies = pd.get_dummies([user_data['marital_status']], prefix='marital_status', drop_first=True)

    # Concatenate the dummies to the user_data_encoded DataFrame
    user_data_encoded = pd.concat([user_data_encoded, education_dummies, employment_dummies, marital_status_dummies], axis=1)

    # Reindex to ensure all columns are present and fill missing ones with 0
    user_data_encoded = user_data_encoded.reindex(columns=feature_names, fill_value=0)

    # Convert columns to correct data types (int for binary features)
    user_data_encoded = user_data_encoded.astype(float)  # Ensure all features are float for XGBoost compatibility

    return user_data_encoded

def recommend_resources(probability):
    if probability >= 0.8:
        st.write("### **Recommendation:**")
        st.write("Given the high likelihood of domestic violence, we recommend the following resources:")
        st.write("- National Domestic Violence Hotline: 1-800-799-7233")
        st.write("- Local women’s shelters and support centers.")
        st.write("- Consider contacting a social worker or legal advisor for assistance.")
    elif probability >= 0.5:
        st.write("### **Recommendation:**")
        st.write("There is a moderate risk of domestic violence. Consider talking to a counselor or seeking advice from a local women’s organization.")
    else:
        st.write("### **Recommendation:**")
        st.write("The risk is low, but it's important to stay informed. Keep track of any potential warning signs.")

def main():
    st.title("AI-Powered Solution for Domestic Violence Prevention")
    st.subheader("Predicting the likelihood of domestic violence based on socio-economic factors.")

    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    education = st.selectbox('Education Level', ('Primary', 'Secondary', 'Tertiary'))
    marital_status = st.selectbox('Marital Status', ('Single', 'Married', 'Divorced'))
    employment = st.selectbox('Employment Status', ('Employed', 'Unemployed', 'Semi-employed'))
    income = st.number_input('Income', min_value=0, max_value=100000, value=20000)

    user_data = {
        'age': age,
        'education': education,
        'marital_status': marital_status,
        'employment': employment,
        'income': income,
    }

    user_data_encoded = encode_user_input(user_data)

    model_choice = st.sidebar.selectbox('Model', ('Random Forest', 'XGBoost', 'Gradient Boosting'))

    if model_choice == 'Random Forest':
        random_forest = joblib.load('random_forest_model.pkl')
        prediction_proba = random_forest.predict_proba(user_data_encoded)[:, 1]
    elif model_choice == 'XGBoost':
        xgboost_model = joblib.load('xgboost_model.pkl')
        prediction_proba = xgboost_model.predict_proba(user_data_encoded)[:, 1]
    elif model_choice == 'Gradient Boosting':
        gradient_boosting_pipeline = joblib.load('gradient_boosting_pipeline.pkl')  # Load the fitted pipeline
        prediction_proba = gradient_boosting_pipeline.predict_proba(user_data_encoded)[:, 1]

    st.write(f"The model predicts a **{prediction_proba[0]*100:.2f}%** likelihood of domestic violence.")

    # Call the recommend_resources function based on prediction probability
    recommend_resources(prediction_proba[0])

if __name__ == '__main__':
    main()
