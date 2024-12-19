import streamlit as st
import pandas as pd
import pickle
import bcrypt
import os


# File to store user data
USER_DATA_FILE = "models/user_data.pkl"

# Load pre-trained models
with open("models/Random Forest_model.pkl", "rb") as rf_model_file:
    random_forest = pickle.load(rf_model_file)
with open("models/XGBoost_model.pkl", "rb") as xgb_model_file:
    xgboost = pickle.load(xgb_model_file)
with open("models/Gradient Boosting_model.pkl", "rb") as gb_model_file:
    gradient_boosting = pickle.load(gb_model_file)

# Model selection dictionary
models = {
    "Random Forest": random_forest,
    "XGBoost": xgboost,
    "Gradient Boosting": gradient_boosting,
}

# Load or initialize user data
if os.path.exists(USER_DATA_FILE):
    with open(USER_DATA_FILE, "rb") as f:
        user_data = pickle.load(f)
else:
    user_data = {}


def save_user_data():
    with open(USER_DATA_FILE, "wb") as f:
        pickle.dump(user_data, f)


# Password hashing functions
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())


def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)


# Registration function
def register_user(email, password):
    if email in user_data:
        return "Email already registered. Please log in."
    user_data[email] = hash_password(password)
    save_user_data()
    return "Registration successful! Please log in."


# Login function
def login_user(email, password):
    if email in user_data and check_password(password, user_data[email]):
        st.session_state["authenticated"] = True
        st.session_state["user_email"] = email
        return "Login successful!"
    return "Invalid email or password."


# Logout function
def logout_user():
    st.session_state["authenticated"] = False
    st.session_state["user_email"] = None


# Check for logout and rerun only if logged in
if (
    st.sidebar.button("Logout", key="logout_button")
    and st.session_state["authenticated"]
):
    logout_user()
    try:
        st.experimental_rerun()
    except:
        pass  # Prevents rerun error from interrupting the app

# Check authentication state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False


# Sidebar for navigation and authentication
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Go to",
    [
        "Overview",
        "Dataset Description",
        "EDA",
        "Model Prediction",
        "Model Evaluation",
        "Team",
        "Contact",
    ],
)

# Login and Registration in Sidebar
if not st.session_state["authenticated"]:
    st.sidebar.subheader("Register or Sign In")

    # Registration block
    with st.sidebar.expander("Register"):
        email = st.text_input("Email", key="register_email")
        password = st.text_input("Password", type="password", key="register_password")
        if st.button("Register", key="register_button"):
            reg_msg = register_user(email, password)
            if "successful" in reg_msg:
                st.sidebar.success(reg_msg)
            else:
                st.sidebar.error(reg_msg)

    # Login block
    with st.sidebar.expander("Sign In"):
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Sign In", key="login_button"):
            login_msg = login_user(email, password)
            if "successful" in login_msg:
                st.sidebar.success(login_msg)
            else:
                st.sidebar.error(login_msg)
else:
    # Logged-in user display and logout option
    st.sidebar.write(f"Welcome, {st.session_state['user_email']}!")

st.markdown(
    """
    <style>
    /* Main app container */
    .stApp {
        background-color: #DEDEDE; /* Darker grey */
    }

    /* Overlay for better readability */
    .main {
        background-color: rgba(255, 255, 255, 0.9); /* Slightly opaque white */
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        color: #333333; /* Darker text */
    }

    /* Set font colors and styles */
    body, .stText {
        color: #333333; /* Default text color */
        font-family: Arial, sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #FF66B2; /* Pink for headings */
        font-family: 'Helvetica', sans-serif;
    }

    label {
        color: #333333 !important; /* Label text color */
    }

    /* Buttons styling */
    .stButton button {
        background-color: #FF66B2;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }

    .stButton button:hover {
        background-color: #D147A3;
        color: white;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #FF66B2;
        color: white;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] h4, 
    section[data-testid="stSidebar"] h5, 
    section[data-testid="stSidebar"] h6 {
        color: white !important; /* Ensure sidebar headings are white */
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Content based on sidebar selection
if page == "Overview":
    st.title("Domestic Violence Risk Prediction and Recommendation App")
    st.write(
        """
        Welcome to the Domestic Violence Risk Prediction and Recommendation App. This tool leverages socio-economic data 
        to assess an individual’s risk of experiencing domestic violence. By analyzing factors such as age, income, 
        education level, employment status, and marital status, the app can provide insights into potential risk levels.

        
        ### How to Use the App:
        Use the sidebar to explore various sections:
        - **Dataset Description**: Information about the data and variables used in the analysis.
        - **Exploratory Data Analysis (EDA)**: Visualizations to help understand data patterns.
        - **Model Prediction**: Enter data to predict an individual's risk level.
        - **Model Evaluation**: View metrics on model performance.
        - **Team & Contact**: Learn more about the project team and how to get in touch.

        This app aims to assist individuals, healthcare providers, and community organizations in identifying and addressing 
        domestic violence risk factors more effectively.
    """
    )

    st.write("Use the sidebar to explore different sections of the app.")

elif page == "Dataset Description":
    st.title("Dataset Description")

    st.write(
        """
        This dataset supports AI models to predict domestic violence risk factors and inform prevention strategies. It was collected 
        through surveys in rural communities, focusing on socio-economic and demographic information aligned with global efforts 
        to address gender equity challenges.
    """
    )

    st.subheader("Data Overview")
    st.write(
        """
        - **Size**: 347 records
        - **Scope**: Socio-economic and demographic factors for women, primarily from developing regions.
        - **Types**: Both numerical and categorical variables.
    """
    )

    st.subheader("Columns")
    st.write(
        """
        - **SL. No**: Unique record ID
        - **Age**: Respondent’s age (numerical)
        - **Education**: Education level (categorical)
        - **Employment**: Employment status (categorical)
        - **Income**: Monthly income (numerical)
        - **Marital Status**: Marital status (categorical)
        - **Violence**: Experience of domestic violence (Yes/No)
    """
    )

    st.subheader("Purpose")
    st.write(
        """
        The data helps identify risk patterns and guide recommendations for support resources to reduce domestic violence.
    """
    )

elif page == "EDA":
    st.title("Exploratory Data Analysis (EDA)")
    st.write(
        """
        This section provides visual insights into key patterns and relationships within the dataset. These visualizations 
        help identify significant factors associated with domestic violence risk, informing model development and targeted interventions.
    """
    )

    # Display EDA charts with descriptions
    st.subheader("Key Insights from Data")
    st.write(
        "Below are some visualizations that highlight important distributions and relationships within the dataset:"
    )

    # Insert EDA1 image (e.g., distribution of age or income)
    st.image("Images/EDA3.png", caption="Distribution of Age Groups or Income Levels")

    # Insert EDA2 image (e.g., correlation between socio-economic factors and domestic violence)
    st.image(
        "Images/EDA1.png",
        caption="Correlation Between Socio-Economic Factors and Domestic Violence Risk",
    )

    # Insert the new EDA3 image (e.g., effect of marital status on violence risk)
    st.image(
        "Images/EDA2.png", caption="Impact of Marital Status on Domestic Violence Risk"
    )

    # Insert the new EDA3 image (e.g., effect of marital status on violence risk)
    st.image(
        "Images/EDA4.png", caption="Impact of Marital Status on Domestic Violence Risk"
    )

    # Additional EDA Insights (if applicable)
    st.write(
        """
        - **Age and Risk**: Certain age groups exhibit a higher risk profile.
        - **Income and Risk**: Higher income levels tend to correlate with a reduced risk.
        - **Employment Status and Risk**: Employment status shows a noticeable impact on risk levels.
        - **Marital Status and Risk**: Individuals in certain marital statuses demonstrate different risk levels for domestic violence.
    """
    )

    st.write(
        "These insights aid in identifying priority risk factors and guiding model development for more effective recommendations."
    )


elif page == "Model Prediction":
    if st.session_state["authenticated"]:
        st.title("Predict Domestic Violence Risk")
        st.write(
            "Enter socio-economic factors to predict the risk of experiencing domestic violence."
        )

        # Model selection
        st.subheader("Select a Model for Prediction")
        model_choice = st.selectbox("Choose a model", list(models.keys()))
        selected_model = models[model_choice]

        # User input collection
        st.subheader("Enter Socio-Economic Factors for Prediction")
        age = st.slider("Age", 15, 100, step=1)
        income_level = st.slider("Income Level", 0, 10000, step=500)
        education = st.selectbox(
            "Education Level", ["Primary", "Secondary", "Tertiary"]
        )
        employment_status = st.selectbox(
            "Employment Status", ["Employed", "Semi-employed", "Unemployed"]
        )
        marital_status = st.selectbox("Marital Status", ["Married", "Unmarried"])

        # Prepare the user input as a DataFrame
        user_data = pd.DataFrame(
            [[age, income_level, education, employment_status, marital_status]],
            columns=["age", "income", "education", "employment", "marital_status"],
        )

        # One-hot encode the user data to match the training data format
        user_data_encoded = pd.get_dummies(
            user_data,
            columns=["education", "employment", "marital_status"],
            drop_first=True,
        )
        # Ensure all required columns are present
        required_columns = [
            "age",
            "income",
            "education_primary",
            "education_secondary",
            "education_tertiary",
            "employment_employed",
            "employment_semi employed",
            "employment_unemployed",
            "marital_status_unmarried",
        ]
        for col in required_columns:
            if col not in user_data_encoded:
                user_data_encoded[col] = (
                    False  # Default to False for missing one-hot encoded columns
                )

        # Reorder columns to match training data
        user_data_encoded = user_data_encoded[required_columns]

        # Prediction and Recommendations
        if st.button("Predict 💬"):
            # Get prediction probabilities instead of just class labels
            prediction_proba = selected_model.predict_proba(user_data_encoded)

            # Apply 0.5 threshold to decide "at risk" (1) or "not at risk" (0)
            if prediction_proba[0][1] >= 0.3:  # Probability of class 1 (at risk)
                st.write(
                    "🚨 The model predicts that this individual is at risk of experiencing domestic violence."
                )
                st.write("**Recommendations:**")
                st.write("- 💼 Access to economic support programs.")
                st.write("- 💍 Marriage or relationship counseling resources.")
                st.write("- 🤝 Community support groups and legal assistance.")
            else:
                st.write(
                    "✅ The model predicts that this individual is not at high risk of experiencing domestic violence."
                )

    else:
        st.error("Please log in to access the prediction feature.")

elif page == "Model Evaluation":
    st.title("Model Evaluation")
    st.write(
        """
        This section presents the accuracy metrics for each model in predicting domestic violence risk after hyperparameter tuning. Accuracy provides a clear measure of each model's overall effectiveness.
    """
    )

    # Display model evaluation metrics (add text or tables with evaluation metrics if available)
    st.subheader("Performance Metrics")
    st.write(
        """
        - **Random Forest**: Accuracy: 76%
        - **XGBoost**: Accuracy: 73%
        - **Gradient Boosting**: Accuracy: 72%
    """
    )

    # Insert EDA5 image for model evaluation (e.g., confusion matrix or ROC curve)
    st.image(
        "Images/EDA5.png", caption="Model Evaluation - Confusion Matrix or ROC Curve"
    )

    # Additional explanation (if applicable)
    st.write(
        """
        - **Model Comparison**: Highlights the performance strengths and limitations of each model, aiding in selecting the best model for risk prediction.
    """
    )

    st.write(
        "Model Evaluation helps us understand each model's predictive capability and guide improvements for future iterations."
    )


elif page == "Team":
    st.title("Meet the Team")

    st.subheader("Nozipho Sithembiso Ndebele")
    st.write(
        """
        **Role**: Data Science Student   
        **Contributions**: Nozipho designed, developed, and implemented the Domestic Violence Risk Prediction 
        and Recommendation App. Her responsibilities included data analysis, model development, and building the 
        application interface. Nozipho’s commitment to addressing social issues through data-driven solutions 
        motivated her to create this app, which aims to identify and mitigate factors contributing to domestic violence risk.
    """
    )

    st.write(
        "This app is a solo project by Nozipho Sithembiso Ndebele, driven by a commitment to support gender equity initiatives."
    )

elif page == "Contact":
    st.title("Get in Touch")
    st.write("Feel free to reach out to us with any questions or feedback.")
    st.text_input("Your Name")
    st.text_input("Your Email")
    st.text_area("Message")
    if st.button("Send Message"):
        st.success("Thank you for reaching out! We'll get back to you soon.")
