Domestic Violence Prediction and Analysis Project
Streamlit App

Table of Contents
Project Overview
Dataset
Packages
Environment Setup
MLflow Integration
Streamlit Deployment
Team Members
1. Project Overview
Our team developed a data-driven solution for predicting and analyzing domestic violence risk, particularly among women in rural areas. This project aims to support interventions by understanding key socio-economic factors contributing to domestic violence.

Steps Followed:

Data Loading: Sourced and loaded datasets covering demographics, employment, education, and other socio-economic variables.
Preprocessing: Cleaned and prepared the data, focusing on variables influencing domestic violence.
Model Training: Developed and evaluated multiple classification models (Logistic Regression, Decision Trees, Random Forests, Gradient Boosting, SVM, XGBoost, and Neural Networks) to predict the likelihood of domestic violence based on socio-economic factors.
Evaluation: Evaluated models based on accuracy and other metrics to identify the best-performing model.
Deployment: Deployed the final model in an interactive web app using Streamlit for easy access and exploration.
This project ultimately aims to provide actionable insights for policymakers and organizations focused on combating domestic violence.

2. Dataset
The dataset consists of socio-economic information sourced from the National Income Dynamics Study (NIDS) Wave 5 and other demographic studies.

Dataset Features:

Column	Description
age	Age of the individual
education	Education level attained
employment	Employment status
income	Monthly or annual income level
marital_status	Marital status (single, married, divorced, etc.)
violence_yes	Target variable indicating occurrence of domestic violence
Additional columns include demographic, household, and socio-economic details.	
3. Packages
The following packages are required to run this project:

Python: 3.11.8
Pandas: 2.2.2
Numpy: 1.26
Scikit-learn: 1.2.2
Matplotlib: 3.8.4
Seaborn: 0.12.2
NLTK: 3.8.1
Surprise: 1.1.1
IPython: 8.20.0
IPython SQL: 0.3.9
PyMySQL: 1.0.2
MLflow: 2.14.1
Streamlit: Latest version
4. Environment Setup
Follow these steps to create the environment and install dependencies.

Creating the Environment:
Create a new Conda environment:
bash
Copy code
conda create --name dv_analysis python=3.11.8
Activate the environment:
bash
Copy code
conda activate dv_analysis
Install packages using pip:
bash
Copy code
pip install -r requirements.txt
Launching the Environment:
Once the environment is set, open it in Jupyter Notebook for interactive analysis:
bash
Copy code
jupyter notebook
5. MLflow Integration
MLflow helps in tracking experiments and managing model lifecycles, facilitating collaboration and reproducibility. Key tasks include:

Logging Hyperparameters and Metrics: Track hyperparameters, accuracy, precision, recall, and other evaluation metrics.
Comparing Models: MLflow’s UI allows easy comparison between model runs to identify the best-performing model.
Refer to the MLflow Quickstart Guide for setup instructions.

6. Streamlit Deployment
Streamlit is a powerful, easy-to-use framework for deploying machine learning models as web apps.

Running Locally:
To set up and run the app locally:

Install Streamlit and other required libraries:
bash
Copy code
pip install -U streamlit numpy pandas scikit-learn
Navigate to the project folder:
bash
Copy code
cd Domestic_Violence_Prediction/
Start the Streamlit app:
bash
Copy code
streamlit run app.py
Upon successful initialization, Streamlit will display a local URL. Visit this URL in your browser to interact with the app.
Deployment:
To deploy the app online:

Set up a Streamlit Cloud or other web hosting account.
Follow Streamlit's deployment instructions.
7. Team Members
Name	Email
Nozipho Ndebele	nozihstheh@gmail.com
License
MIT License

Resources
MLflow Documentation: MLflow
Streamlit Documentation: Streamlit