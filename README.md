# Domestic Violence Prediction and Analysis Project

![](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white) [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](URL_TO_YOUR_APP)

<div style="text-align: center;">
<img src="https://slamatlaw.co.za/wp-content/uploads/2021/03/domestic-violence.jpg" alt="Anime Image" width="800"/>
</div>

## Table of contents
* [1. Project Overview](#project-description)
* [2. Dataset](#dataset)
* [3. Packages](#packages)
* [4. Environment](#environment)
* [5. Team Members](#team-members)
* [6. MLflow Integration](#MLflow-Integration)
* [7. Streamlit Deployment](#Streamlit-Deployment)

## 1. Project Overview <a class="anchor" id="project-description"></a>
Our team developed a data-driven solution for predicting and analyzing domestic violence risk, particularly among women in rural areas. This project aims to support interventions by understanding key socio-economic factors contributing to domestic violence.

### Steps Followed:

1. Data Loading: Sourced and loaded datasets covering demographics, employment, education, and other socio-economic variables.
Preprocessing: Cleaned and prepared the data, focusing on variables influencing domestic violence.
2. Model Training: Developed and evaluated multiple classification models (Logistic Regression, Decision Trees, Random Forests, Gradient Boosting, SVM, XGBoost, and Neural Networks) to predict the likelihood of domestic violence based on socio-economic factors.
3. Evaluation: Evaluated models based on accuracy and other metrics to identify the best-performing model.
4. Deployment: Deployed the final model in an interactive web app using Streamlit for easy access and exploration.
This project ultimately aims to provide actionable insights for policymakers and organizations focused on combating domestic violence.

## 2. Dataset <a class="anchor" id="dataset"></a>
The dataset consists of socio-economic information sourced from the National Income Dynamics Study (NIDS) Wave 5 and other demographic studies.
	

**Dataset Features:**
| **Column**                                                                                  | **Description**              
|---------------------------------------------------------------------------------------------|--------------------   
| age   | 	Age of the individual
| education | Education level attained
| employment | Employment status
| income | Monthly or annual income level
| marital_status | Marital status (single, married, divorced, etc.)
| violence_yes | Target variable indicating occurrence of domestic violence
| Additional columns | include demographic, household, and socio-economic details.

## 3. Packages <a class="anchor" id="packages"></a>

To carry out all the objectives for this repo, the following necessary dependencies were loaded:

### Streamlit app dependencies:
- `streamlit` - For building the interactive web application.
- `pickle-mixin` - For working with pickled objects.
- `bcrypt` - For securely handling password encryption and authentication.
- `os-sys` - Provides access to miscellaneous operating system interfaces.

### Notebook dependencies:
- `pandas` - For data manipulation and analysis.
- `numpy` - For numerical computations.
- `matplotlib` - For data visualization and plotting.
- `seaborn` - For statistical data visualization.
- `plotly` - For interactive plots and data visualizations.
- `scikit-learn` - For machine learning algorithms and model evaluation.
- `xgboost` - For gradient boosting-based machine learning models.
- `joblib` - For saving and loading models.
- `statsmodels` - For statistical analysis and modeling.
- `mlflow` - For tracking experiments and managing machine learning models.

### Python environment:
- `Python 3.11.8` - The Python version used in this environment.

## 4. Environment <a class="anchor" id="environment"></a>

Use the Anaconda programme to create the environment. Click on the plus icon at the bottom of the Anaconda environments pane. Select All on the drop down menu to select the packages needed to run this report. After selecting the tick boxes - click on apply. Then right click on the play button and scroll down to open in Jupyter notebook to install this report.

### Create the new evironment - you only need to do this once

```bash
# create the conda environment
conda create --name <env>
```

### This is how you activate the virtual environment in a terminal and install the project dependencies

```bash
# activate the virtual environment
conda activate <env>
# install the pip package
conda install pip
# install the requirements for this project
pip install -r requirements.txt
```
## 5. MLFlow<a class="anchor" id="mlflow"></a>

MLOps, which stands for Machine Learning Operations, is a practice focused on managing and streamlining the lifecycle of machine learning models. The modern MLOps tool, MLflow is designed to facilitate collaboration on data projects, enabling teams to track experiments, manage models, and streamline deployment processes. For experimentation, testing, and reproducibility of the machine learning models in this project, you will use MLflow. MLflow will help track hyperparameter tuning by logging and comparing different model configurations. This allows you to easily identify and select the best-performing model based on the logged metrics.

- Please have a look here and follow the instructions: https://www.mlflow.org/docs/2.7.1/quickstart.html#quickstart

## 6. Streamlit<a class="anchor" id="streamlit"></a>

### What is Streamlit?

[Streamlit](https://www.streamlit.io/)  is a framework that acts as a web server with dynamic visuals, multiple responsive pages, and robust deployment of your models.

In its own words:
> Streamlit ... is the easiest way for data scientists and machine learning engineers to create beautiful, performant apps in only a few hours!  All in pure Python. All for free.

> It’s a simple and powerful app model that lets you build rich UIs incredibly quickly.

[Streamlit](https://www.streamlit.io/)  takes away much of the background work needed in order to get a platform which can deploy your models to clients and end users. Meaning that you get to focus on the important stuff (related to the data), and can largely ignore the rest. This will allow you to become a lot more productive.  

##### Description of files

For this repository, we are only concerned with a single file:

| File Name              | Description                       |
| :--------------------- | :--------------------             |
| `base_app.py`          | Streamlit application definition. |


#### 6.1 Running the Streamlit web app on your local machine

As a first step to becoming familiar with our web app's functioning, we recommend setting up a running instance on your own local machine. To do this, follow the steps below by running the given commands within a Git bash (Windows), or terminal (Mac/Linux):

- Ensure that you have the prerequisite Python libraries installed on your local machine:

 ```bash
 pip install -U streamlit numpy pandas scikit-learn
 ```

- Navigate to the base of your repo where your base_app.py is stored, and start the Streamlit app.

 ```bash
 cd Women_and_girl-s_safety/Streamlit/
 streamlit run base_app.py
 ```

 If the web server was able to initialise successfully, the following message should be displayed within your bash/terminal session:

```
  You can now view your Streamlit app in your browser.

    Local URL: http://localhost:8501
    Network URL: http://192.168.43.41:8501
```
You should also be automatically directed to the base page of your web app. This should look something like:

<div id="s_image" align="center">
  <img src="App_picture.png" width="850" height="400" alt=""/>
</div>

Congratulations! You've now officially deployed your first web application!

#### 6.2 Deploying your Streamlit web app

- To deploy your app for all to see, click on `deploy`.
  
- Please note: If it's your first time deploying it will redirect you to set up an account first. Please follow the instructions.

## 7. Team Members<a class="anchor" id="team-members"></a>

| Names                                                                                     | Emails              
|-------------------------------------------------------------------------------------------------------------------------|----------             
| [Nozipho Ndebele](https://github.com/NoziphoNdebele)                                      | nozihstheh@gmail.com

## License
 [MIT](https://choosealicense.com/licenses/mit/#) 