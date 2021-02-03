import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.preprocessing import MinMaxScaler


@st.cache(suppress_st_warning=True)
def load_dataset():
    data_url = "https://raw.githubusercontent.com/budipryg/data_apps/main/heart_app"
    return pd.read_csv(data_url + "/heart.csv")


def main():
    st.sidebar.title("Try it Here!")
    app_mode = st.sidebar.selectbox("Choose the App Mode",
                                    ["Home", "Run Prediction"])
    if app_mode == "Home":
        st.title('Heart Disease Prediction')
        st.markdown("*created by: Budi Prayoga * <a href='https://www.linkedin.com/in/budipryg/'>--> Linkedin</a>",
                    unsafe_allow_html=True)
        st.markdown(
            "***Disclaimer : This application is not intended for practical use***")
        st.subheader('App Description')
        st.markdown("""<div style='text-align: justify'>Technology has become the most important apect in future development, including in the health sector.
        A real example of the use of this technology is to conduct initial screening whether a patient has a certain disease or not based on some data taken.
        This application will predict whether someone has heart disease or not based on the following parameters : </div>""", unsafe_allow_html=True)
        st.markdown("""
            - **age** : age in years
            - **sex** : (1 = male; 0 = female)
            - **cp** : chest pain type (4 values)
            - **trestbps** : resting blood pressure (in mm Hg on admission to the hospital)
            - **chol** : serum cholestoral in mg/dl
            - **fbs** : (fasting blood sugar &gt; 120 mg/dl) (1 = true; 0 = false)
            - **restecg** : resting electrocardiographic results (values 0,1,2)
            - **thalach** : maximum heart rate achieved
            - **exang** : exercise induced angina (1 = yes; 0 = no)
            - **oldpeak** : ST depression induced by exercise relative to rest
            - **slope** : the slope of the peak exercise ST segment
            - **ca** : number of major vessels (0-3) colored by flourosopy
            - **thal** : 3 = normal; 6 = fixed defect; 7 = reversable defect
        """)
        st.subheader('About Dataset')
        st.markdown("""<div style='text-align: justify'><b>Machine Learning</b> Model is used to predict the outcome.
        However, before this model could work, we have to train the model with existing data and label. The dataset is obtained from <em>kaggle.com</em>, one of the largest data science community.
        This dataset contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland dataset is the only one that has been used by ML researchers to
        this date.</div>""", unsafe_allow_html=True)
        st.subheader('Acknowledgements')
        st.markdown("""
            - Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
            - University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
            - University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
            - V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.
        """)
    elif app_mode == "Run Prediction":
        model = st.sidebar.selectbox(
            "Select Model",
            ("Logistic Regression", "Decision Tree",
             "Random Forest", "CatBoost")
        )

        st.sidebar.markdown(
            "***Before making a prediction, feel free to change the model parameter and find the best model at a real time!***")

        # if model == "Deep Learning":
        #     test_size = st.sidebar.slider(
        #         "Test Size (Fraction of Dataset)", min_value=0.1, max_value=0.4, value=0.2)
        #     nn_1 = st.sidebar.slider(
        #         "Number of neuron on 1st layer", min_value=1, max_value=20, value=13)
        #     dl_1 = st.sidebar.slider(
        #         "Dropout layer on 1st layer", min_value=0.1, max_value=0.5, value=0.2)
        #     nn_2 = st.sidebar.slider(
        #         "Number of neuron on 2nd layer", min_value=1, max_value=20, value=6)
        #     dl_2 = st.sidebar.slider(
        #         "Dropout layer on 2nd layer", min_value=0.1, max_value=0.5, value=0.2)
        #     nn_3 = st.sidebar.slider(
        #         "Number of neuron on 3rd layer", min_value=1, max_value=20, value=3)
        #     dl_3 = st.sidebar.slider(
        #         "Dropout layer on 3rd layer", min_value=0.1, max_value=0.5, value=0.2)
        # else:
        test_size = st.sidebar.slider(
            "Test Size (Fraction of Dataset)", min_value=0.1, max_value=0.4, value=0.2)

        eval_res = pd.DataFrame()

        # if model == "Deep Learning":
        #     dl_model(test_size, nn_1, dl_1, nn_2, dl_2, nn_3, dl_3)
        # else:
        eval_res = ml_train_eval(model, test_size)
        st.sidebar.write('Model Evaluation Result')
        st.sidebar.dataframe(eval_res)

        st.header("Make Prediction Here!")
        col1, col2 = st.beta_columns(2)
        d_sex = dict(zip([0, 1], ['male', 'female']))
        d_cp = dict(zip([1, 2, 3, 4], ['typical angina',
                                       'atypical angina', 'non-anginal pain', 'asymptomatic']))
        d_fbs = dict(
            zip([0, 1], ['lower than 120mg/ml', 'greater than 120mg/ml']))
        d_restecg = dict(zip(
            [0, 1, 2], ['normal', 'ST-T wave abnormality', 'left ventricular hypertrophy']))
        d_exang = dict(zip([0, 1], ["no", "yes"]))
        d_slope = dict(zip([1, 2, 3], ['upsloping', 'flat', 'downsloping']))
        d_thal = dict(
            zip([1, 2, 3], ['normal', 'fixed defect', 'reversable defect']))
        with col1:
            age = st.number_input("age (years)", min_value=1,
                                  max_value=120, value=50)
            sex = st.selectbox("sex", (0, 1), format_func=lambda x: d_sex[x])
            cp = st.selectbox("chest_pain_type", (1, 2, 3, 4),
                              format_func=lambda x: d_cp[x])
            trestbps = st.number_input(
                "resting_blood_pressure (mm Hg)", min_value=60, max_value=250, value=120)
            chol = st.number_input("cholesterol (mg/dl)",
                                   min_value=50, max_value=700, value=250)
            fbs = st.selectbox("fasting_blood_sugar",
                               (0, 1), format_func=lambda x: d_fbs[x])
            restecg = st.selectbox(
                "rest_ecg", (0, 1, 2), format_func=lambda x: d_restecg[x])

        with col2:
            thalach = st.number_input(
                "max_heart_rate_achieved", min_value=60, max_value=250, value=150)
            exang = st.selectbox("exercise_induced_angina",
                                 (0, 1), format_func=lambda x: d_exang[x])
            oldpeak = st.number_input(
                "st_depression", min_value=0.00, max_value=10.00, step=0.01, value=0.00)
            slope = st.selectbox(
                "st_slope", (1, 2, 3), format_func=lambda x: d_slope[x])
            ca = st.number_input("num_major_vessels",
                                 min_value=0, max_value=4, value=0)
            thal = st.selectbox(
                "thalassemia", (1, 2, 3), format_func=lambda x: d_thal[x])

        if st.button("Make Prediction"):
            pred_input = np.array([age, sex, cp, trestbps, chol, fbs, restecg,
                                   thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
            st.markdown("Heart Disease is " + "**" +
                        pred_result(model, test_size, pred_input) + "**")


@st.cache(suppress_st_warning=True)
def train_test(test_size):
    df = load_dataset()
    X = df.drop('target', axis=1).values
    y = df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    return [X_train, y_train, X_test, y_test]


@st.cache(suppress_st_warning=True)
def ml_pred(model, pred_input):
    pred = model.predict(pred_input)
    if pred == 0:
        return "Not Present"
    elif pred == 1:
        return "Present"


@st.cache(suppress_st_warning=True)
def pred_result(model, test_size, pred_input):
    split = train_test(test_size)
    X_train, y_train, X_test, y_test = split[0], split[1], split[2], split[3]
    if model == "Logistic Regression":
        return ml_pred(log_reg(X_train, y_train, X_test, y_test), pred_input)
        return X_train
    elif model == "Decision Tree":
        return ml_pred(d_tree(X_train, y_train, X_test, y_test), pred_input)
    elif model == "Random Forest":
        return ml_pred(ran_for(X_train, y_train, X_test, y_test), pred_input)
    elif model == "CatBoost":
        return ml_pred(cat_b(X_train, y_train, X_test, y_test), pred_input)


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def ml_train_eval(model, test_size):
    split = train_test(test_size)
    X_train, y_train, X_test, y_test = split[0], split[1], split[2], split[3]
    if model == "Logistic Regression":
        lr = log_reg(X_train, y_train, X_test, y_test)
        pred = lr.predict(X_test)
        result = pd.DataFrame(precision_recall_fscore_support(y_test, pred), index=[
                              'precision', 'recall', 'fscore', 'support'], columns=['not_present', 'present'])
        return result
    elif model == "Decision Tree":
        dt = d_tree(X_train, y_train, X_test, y_test)
        pred = dt.predict(X_test)
        result = pd.DataFrame(precision_recall_fscore_support(y_test, pred), index=[
                              'precision', 'recall', 'fscore', 'support'], columns=['not_present', 'present'])
        return result
    elif model == "Random Forest":
        rf = ran_for(X_train, y_train, X_test, y_test)
        pred = rf.predict(X_test)
        result = pd.DataFrame(precision_recall_fscore_support(y_test, pred), index=[
            'precision', 'recall', 'fscore', 'support'], columns=['not_present', 'present'])
        return result
    elif model == "CatBoost":
        cb = cat_b(X_train, y_train, X_test, y_test)
        pred = cb.predict(X_test)
        result = pd.DataFrame(precision_recall_fscore_support(y_test, pred), index=[
            'precision', 'recall', 'fscore', 'support'], columns=['not_present', 'present'])
        return result


@st.cache(suppress_st_warning=True)
def log_reg(X_train, y_train, X_test, y_test):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    return lr


@st.cache(suppress_st_warning=True)
def d_tree(X_train, y_train, X_test, y_test):
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    return dt


@st.cache(suppress_st_warning=True)
def ran_for(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    return rf


@st.cache(suppress_st_warning=True)
def cat_b(X_train, y_train, X_test, y_test):
    cb = CatBoostClassifier()
    cb.fit(X_train, y_train)
    return cb


main()


def cache_on_button_press(label, **cache_kwargs):
    """Function decorator to memoize function executions.
    Parameters
    ----------
    label : str
        The label for the button to display prior to running the cached funnction.
    cache_kwargs : Dict[Any, Any]
        Additional parameters (such as show_spinner) to pass into the underlying @st.cache decorator.
    Example
    -------
    This show how you could write a username/password tester:
    >>> @cache_on_button_press('Authenticate')
    ... def authenticate(username, password):
    ...     return username == "buddha" and password == "s4msara"
    ...
    ... username = st.text_input('username')
    ... password = st.text_input('password')
    ...
    ... if authenticate(username, password):
    ...     st.success('Logged in.')
    ... else:
    ...     st.error('Incorrect username or password')
    """
    internal_cache_kwargs = dict(cache_kwargs)
    internal_cache_kwargs['allow_output_mutation'] = True
    internal_cache_kwargs['show_spinner'] = False

    def function_decorator(func):
        @ functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            @ st.cache(**internal_cache_kwargs)
            def get_cache_entry(func, args, kwargs):
                class ButtonCacheEntry:
                    def __init__(self):
                        self.evaluated = False
                        self.return_value = None

                    def evaluate(self):
                        self.evaluated = True
                        self.return_value = func(*args, **kwargs)
                return ButtonCacheEntry()
            cache_entry = get_cache_entry(func, args, kwargs)
            if not cache_entry.evaluated:
                if st.button(label):
                    cache_entry.evaluate()
                else:
                    raise st.stop()
            return cache_entry.return_value
        return wrapped_func
    return function_decorator
