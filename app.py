#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pdfplumber
import nltk
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path


# In[3]:


# Load the job dataset
df = pd.read_csv('job_descriptions.csv')

# Check dataset structure
print(df.head())


# In[4]:


X = df['Job Description']
y = df['Role']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


# Build the pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())


# In[6]:


# Train the model
model.fit(X_train, y_train)


# In[7]:


# Test accuracy
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))


# In[9]:


job_desc_input = input("📌 Enter the job description: ")
predicted_role = model.predict([job_desc_input])[0]
print("🎯 Predicted Job Role:", predicted_role)


# In[13]:


# ✅ Set the Tesseract path correctly
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Sarde\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# ✅ Set the Poppler path (Ensure it's installed)
poppler_path = r"C:\poppler-24.02.0\Library\bin"

# Convert PDF to images
def pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
    return images

# Preprocess image for better OCR
def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

# Extract text using OCR
def extract_text_from_images(images):
    extracted_text = ""
    for img in images:
        preprocessed_img = preprocess_image(img)
        text = pytesseract.image_to_string(preprocessed_img, lang="eng")
        extracted_text += text + "\n"
    return extracted_text

# ✅ Path to your PDF (Ensure the file exists)
pdf_path = "ashwin1.pdf"

# Convert PDF pages to images
images = pdf_to_images(pdf_path)

# Extract text from images
extracted_text = extract_text_from_images(images)

# Print extracted text
print(extracted_text)


# In[15]:


import pandas as pd
import re
from fuzzywuzzy import process
import spacy

# Load spaCy's Named Entity Recognition (NER) model
nlp = spacy.load("en_core_web_sm")


# Extract skills column and convert to a set for fast lookup
skills_list = set(df["skills"].dropna().str.lower().tolist())

# Function to clean and preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # Remove special characters
    return text

# Function to extract skills using keyword matching & fuzzy matching
def extract_skills(text):
    cleaned_text = preprocess_text(text)
    extracted_skills = set()

    # Direct keyword matching
    for skill in skills_list:
        if skill in cleaned_text:
            extracted_skills.add(skill)

    # Fuzzy matching for variations
    for word in cleaned_text.split():
        match, score = process.extractOne(word, skills_list)
        if score > 85:  # Adjust threshold if needed
            extracted_skills.add(match)

    return list(extracted_skills)

# Function to extract skills using Named Entity Recognition (NER)
def extract_skills_ner(text):
    doc = nlp(text)
    extracted_skills = [ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART"]]
    return extracted_skills

# Extract skills using both methods
skills_matched = extract_skills(extracted_text)
skills_ner = extract_skills_ner(extracted_text)

# Combine results from both methods
final_skills = list(set(skills_matched + skills_ner))

# Print extracted skills
print("Extracted Skills:", final_skills)


# In[17]:


# ✅ Ensure column names are formatted properly
df.columns = df.columns.str.strip().str.lower()

# ✅ Standardize job roles in the dataset for comparison
df["role"] = df["role"].str.strip().str.lower()

# ✅ Function to get skills for a predicted job role
def get_job_skills(predicted_job, df):
    """
    Extracts job-related skills from the dataset based on the predicted job role.
    
    :param predicted_job: The job title predicted by the model.
    :param df: The Pandas DataFrame containing job roles and skills.
    :return: List of skills required for the predicted job.
    """
    predicted_job = predicted_job.strip().lower()  # Ensure matching
    job_row = df[df["role"] == predicted_job]  # Exact match after standardization
    
    if not job_row.empty:
        skills_str = job_row.iloc[0]["skills"]
        if isinstance(skills_str, str):  # Ensure it's not NaN
            return [skill.strip() for skill in skills_str.split(",") if skill.strip()]
    
    return []  # Return empty list if no match found

# ✅ Function to calculate skill match percentage
def calculate_match_percentage(resume_skills, job_skills):
    """
    Compares extracted resume skills with job-required skills and calculates the match percentage.
    
    :param resume_skills: List of extracted skills from the resume.
    :param job_skills: List of required skills for the job role.
    :return: Match percentage (0-100), matched skills, and missing skills.
    """
    resume_skills = set(skill.lower().strip() for skill in resume_skills)
    job_skills = set(skill.lower().strip() for skill in job_skills)

    matched_skills = resume_skills.intersection(job_skills)
    missing_skills = job_skills - resume_skills

    match_percentage = (len(matched_skills) / len(job_skills) * 100) if job_skills else 0

    return round(match_percentage, 2), matched_skills, missing_skills


# ✅ Predicted job role (From ML model)
predicted_job = predicted_role  # Assuming `predicted_role` is already defined

# ✅ Extract job skills based on the predicted role
job_skills = get_job_skills(predicted_job, df)

# ✅ Resume skills (Extracted separately)
resume_skills = final_skills  # Assuming `final_skills` contains extracted skills from the resume

# ✅ Compare skills and calculate match percentage
match_percentage, matched_skills, missing_skills = calculate_match_percentage(resume_skills, job_skills)



# In[18]:


# ✅ Debugging: Print extracted skills
print(f"📌 Extracted Job Skills: {job_skills}")
print(f"📌 Extracted Resume Skills: {resume_skills}")

# ✅ Proceed to match skills
match_percentage, matched_skills, missing_skills = calculate_match_percentage(resume_skills, job_skills)


# In[19]:


import re
from fuzzywuzzy import process

def preprocess_text(text):
    """ Lowercase, remove special characters, and normalize spaces """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation
    return text.strip()

def tokenize_skills(skill_set):
    """ Tokenize a set of skills by splitting multi-word phrases """
    tokenized_skills = set()
    for skill in skill_set:
        tokenized_skills.update(preprocess_text(skill).split())  # Break down into words
    return tokenized_skills

def fuzzy_match_skills(resume_skills, job_skills, threshold=85):
    """ Perform fuzzy matching to account for skill variations """
    matched_skills = set()
    for resume_skill in resume_skills:
        match, score = process.extractOne(resume_skill, job_skills)
        if score >= threshold:
            matched_skills.add(match)
    return matched_skills

# ✅ Convert job skills (ensures it's properly formatted)
job_skills = tokenize_skills(job_skills)  # Convert into a set of words

# ✅ Convert resume skills into a set of words
normalized_resume_skills = tokenize_skills(resume_skills)

# ✅ Apply fuzzy matching to handle minor variations
matched_skills = fuzzy_match_skills(normalized_resume_skills, job_skills)

# ✅ Calculate match percentage
missing_skills = job_skills - matched_skills
match_percentage = (len(matched_skills) / len(job_skills) * 100) if job_skills else 0

# ✅ Print the updated output
print(f"🎯 Predicted Job Role: {predicted_job}")
print(f"✅ Match Percentage: {match_percentage:.2f}%")
print(f"🟢 Matched Skills: {matched_skills}")
print(f"🔴 Missing Skills: {missing_skills}")


# In[ ]:




