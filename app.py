import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="AI Resume Screening", page_icon="📄")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "resume_screening_model.pkl"
COLUMNS_PATH = BASE_DIR / "resume_feature_columns.pkl"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, list(columns)

st.title("AI Resume Screening")
st.write("Enter candidate details to estimate the probability of shortlisting.")

try:
    model, feature_columns = load_model()
except Exception as e:
    st.error("Model files could not be loaded. Keep both .pkl files in the same folder as app.py.")
    st.stop()

with st.form("resume_form"):
    years_experience = st.number_input("Years of experience", min_value=0, max_value=50, value=2)
    skills_match_score = st.slider("Skills match score", min_value=0, max_value=100, value=75)
    education_level = st.selectbox(
        "Education level",
        ["Bachelors", "Masters", "High School", "PhD"]
    )
    project_count = st.number_input("Number of relevant projects", min_value=0, max_value=100, value=3)
    resume_length = st.number_input("Resume length (words)", min_value=1, max_value=10000, value=450)
    github_activity = st.number_input("GitHub activity", min_value=0, max_value=10000, value=50)

    submitted = st.form_submit_button("Check Shortlisting Probability")

if submitted:
    education_mapping = {
        "Bachelors": 0,
        "Masters": 1,
        "High School": 2,
        "PhD": 3
    }

    candidate = pd.DataFrame([{
        "years_experience": years_experience,
        "skills_match_score": float(skills_match_score),
        "education_level": education_mapping[education_level],
        "project_count": project_count,
        "resume_length": resume_length,
        "github_activity": github_activity
    }]).reindex(columns=feature_columns, fill_value=0)

    try:
        probability = float(model.predict_proba(candidate)[0][1])
        st.subheader("Estimated Shortlisting Probability")
        st.metric("Probability", f"{probability:.2%}")
        st.progress(max(0.0, min(1.0, probability)))
        st.caption("This is an estimate from the trained model, not a final hiring decision.")
    except Exception as e:
        st.error(f"Could not generate prediction: {e}")
