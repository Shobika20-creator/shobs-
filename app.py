import streamlit as st
import pandas as pd
import joblib
import pdfplumber
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path
import spacy
from fuzzywuzzy import process
import re
import tempfile  

st.cache_data.clear()

# ✅ Load Pre-trained Model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")
      # Ensure 'model.pkl' is available

model = load_model()

# ✅ Load Job Dataset
df = pd.read_csv("job_descriptions.csv")

# ✅ Set the Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Sarde\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
poppler_path = r"C:\poppler-24.02.0\Library\bin"

# ✅ Load NLP Model
nlp = spacy.load("en_core_web_sm")

# ✅ Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

# ✅ Extract skills from dataset
skills_list = set(df["skills"].dropna().str.lower().tolist())

# ✅ Extract skills using keyword & fuzzy matching
def extract_skills(text):
    cleaned_text = preprocess_text(text)
    extracted_skills = set()
    
    for skill in skills_list:
        if skill in cleaned_text:
            extracted_skills.add(skill)

    for word in cleaned_text.split():
        match, score = process.extractOne(word, skills_list)
        if score > 85:
            extracted_skills.add(match)

    return list(extracted_skills)

# ✅ Extract text from uploaded PDF

def extract_text_from_pdf(uploaded_file):
    # Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.getvalue())  # Save the uploaded file content
        temp_pdf_path = temp_pdf.name  # Get the file path
    
    # Convert PDF to images
    images = convert_from_path(temp_pdf_path, dpi=300, poppler_path=poppler_path)
    
    # Extract text from each image
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    
    return text


# ✅ Get skills required for a job
def get_job_skills(predicted_job, df):
    df.columns = df.columns.str.strip().str.lower()  # Normalize column names
    
    if "role" not in df.columns:
        print("Error: 'role' column not found! Available columns:", df.columns)
        return []  # Return empty list if column is missing
    
    job_row = df[df["role"].str.lower() == predicted_job.lower()]
    
    if job_row.empty:
        print(f"Error: No matching role found for {predicted_job}")
        return []
    
    return job_row["skills"].values[0].split(",")  # Assuming skills are comma-separated


# ✅ Calculate skill match percentage
def calculate_match_percentage(resume_skills, job_skills):
    resume_skills = set(resume_skills)
    job_skills = set(job_skills)
    matched_skills = resume_skills.intersection(job_skills)
    missing_skills = job_skills - resume_skills
    match_percentage = (len(matched_skills) / len(job_skills) * 100) if job_skills else 0
    return round(match_percentage, 2), matched_skills, missing_skills

# ✅ Streamlit UI
st.title("🔍 AI-Powered Job Role Prediction & Skill Matcher")
st.write("Upload a resume PDF or enter a job description to predict the job role and skill match percentage.")

# ✅ Text input for job description
job_desc_input = st.text_area("📌 Enter Job Description:")

# ✅ File upload for resume
uploaded_file = st.file_uploader("📂 Upload Resume (PDF only)", type=["pdf"])

# ✅ Process the input
if st.button("Predict Job Role"):
    if job_desc_input.strip() or uploaded_file:
        if uploaded_file:
            extracted_text = extract_text_from_pdf(uploaded_file)
            st.text_area("📄 Extracted Resume Text:", extracted_text, height=200)
        else:
            extracted_text = job_desc_input
        
        # ✅ Predict job role
        predicted_role = model.predict([extracted_text])[0]
        st.success(f"🎯 Predicted Job Role: {predicted_role}")

        # ✅ Get required skills
        job_skills = get_job_skills(predicted_role, df)
        extracted_resume_skills = extract_skills(extracted_text)

        # ✅ Match percentage
        match_percentage, matched_skills, missing_skills = calculate_match_percentage(extracted_resume_skills, job_skills)

        st.info(f"📊 Match Percentage: {match_percentage}%")
        st.write(f"✅ **Matched Skills:** {', '.join(matched_skills) if matched_skills else 'None'}")
        st.write(f"⚠️ **Missing Skills:** {', '.join(missing_skills) if missing_skills else 'None'}")

    else:
        st.error("⚠️ Please enter a job description or upload a resume.")








