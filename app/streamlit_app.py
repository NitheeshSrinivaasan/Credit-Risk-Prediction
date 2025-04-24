import streamlit as st
import pandas as pd
import joblib

# Load pre-trained model and preprocessing tools
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
features = joblib.load("model/features.pkl")


# Set page config
st.set_page_config(page_title="Credit Risk Prediction", layout="wide")

st.title("Credit Risk Prediction App")

# Sidebar Navigation
menu = st.sidebar.radio("Menu", ["Home", "EDA", "Predict"])

# Load the original dataset (for EDA)
@st.cache_data
def load_data():
    df = pd.read_csv("data/german_credit_data.csv")
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
    df['Checking account'] = df['Checking account'].fillna('unknown')
    return df

df = load_data()

if menu == "Home":
    st.markdown("### Project Overview")
    st.write("""
    This application allows users to:
    - Explore insights from the German Credit dataset.
    - Predict whether a loan applicant is a good or bad credit risk.
    - Visualize model explanations using SHAP values.
    """)
    st.image("outputs/rel_credit_duration.png", caption="Credit Amount vs Duration by Risk", use_container_width=True)
    st.image("outputs/rel_age_credit.png", caption="Age vs Credit Amount by Risk", use_container_width=True)
    st.image("outputs/correlation_heatmap.png", caption="Correlation Heatmap", use_container_width=True)

elif menu == "EDA":
    st.header("Exploratory Data Analysis")
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.write("### Purpose vs Risk")
    st.image("outputs/rel_purpose_risk.png", use_container_width=True)

    st.write("### Saving Account vs Risk")
    st.image("outputs/rel_saving_risk.png", use_container_width=True)

    st.write("### Feature Importance (from best model)")
    st.image("outputs/feature_importance.png", use_container_width=True)

    st.write("### ROC Curve Comparison")
    st.image("outputs/roc_comparison.png", use_container_width=True)

# User input section
elif menu == "Predict":
    st.header("Predict Credit Risk")
    def user_input():
        age = st.number_input("Age", min_value=18, max_value=100)
        sex = st.selectbox("Sex", ["male", "female"])
        job = st.selectbox("Job", [0, 1, 2, 3])
        housing = st.selectbox("Housing", ["own", "rent", "free"])
        saving_accounts = st.selectbox("Saving accounts", ["little", "moderate", "quite rich", "rich"])
        checking_account = st.selectbox("Checking account", ["little", "moderate", "quite rich", "rich"])
        credit_amount = st.number_input("Credit amount (in DM)", min_value=0.0)
        duration = st.number_input("Duration (in months)", min_value=1)
        purpose = st.selectbox("Purpose", ["car", "furniture/equipment", "radio/TV", "domestic appliances",
                                       "repairs", "education", "business", "vacation/others"])

        data = {
            "Age": age,
            "Sex": sex,
            "Job": job,
            "Housing": housing,
            "Saving accounts": saving_accounts,
            "Checking account": checking_account,
            "Credit amount": credit_amount,
            "Duration": duration,
            "Purpose": purpose
        }

        return pd.DataFrame([data])

    input_df = user_input()

    # Add Predict button
    if st.button("Predict"):
        # Apply same label encoding as in training
        label_mappings = {
            "Sex": {"male": 0, "female": 1},
            "Housing": {"own": 0, "rent": 1, "free": 2},
            "Saving accounts": {"little": 0, "moderate": 1, "quite rich": 2, "rich": 3},
            "Checking account": {"little": 0, "moderate": 1, "quite rich": 2, "rich": 3},
            "Purpose": {
                "car": 0, "furniture/equipment": 1, "radio/TV": 2, "domestic appliances": 3,
                "repairs": 4, "education": 5, "business": 6, "vacation/others": 7
            }
        }

        for col, mapping in label_mappings.items():
            if col in input_df:
                input_df[col] = input_df[col].map(mapping)

        # Align columns
        input_df = input_df[features]

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        # Output
        st.subheader("Prediction")
        st.write("Good Credit Risk" if prediction == 1 else "Bad Credit Risk")

        st.subheader("Confidence")
        st.write(f"Probability of good credit: {probability:.2f}")
