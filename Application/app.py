import os
import streamlit as st
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

import pdfplumber
from io import BytesIO

# Function to load CSS
def load_css(file_name):
    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)

    # Construct the absolute path to the CSS file
    file_path = os.path.join(script_dir, file_name)

    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the CSS
load_css('styles.css')

def get_recommendations(text, gender, experience, age, language):
    if language == 'Swedish':
        prompt = f"{text}\n\nGivet att den ideala kandidaten är {employment_type}, {gender}, {experience}, {age}, {location}, {driving_license} och {education}, hur kan denna jobbannons förbättras?"
        prompt = f"{text}\n\nJag har en jobbannons och jag vill förbättra den baserat på vissa kriterier. Den ideala kanditaten för min jobbannons har följande egenskaper: {employment_type}, {gender}, {experience}, {age}, {location}, {driving_license} och {education}, hur kan min jobbannons förbättras för att bättre attrahera den ideala kandidaten? Skriv svaret på Svenska."
        system_message = "Du är en hjälpsam assistent."
    else:  # Default to English
        prompt = f"{text}\n\nGiven that the ideal candidate is {employment_type}, {gender}, {experience}, {age}, {location}, {driving_license} och {education}, how could this job posting be improved?"
        prompt = f"{text}\n\nJag har en jobbannons och jag vill förbättra den baserat på vissa kriterier. Den ideala kanditaten för min jobbannons har följande egenskaper: {employment_type}, {gender}, {experience}, {age}, {location}, {driving_license} och {education}, hur kan min jobbannons förbättras för att bättre attrahera den ideala kandidaten? Skriv svaret på Engelska."
        system_message = "You are a helpful assistant."

    response = client.completions.create(model="gpt-3.5-turbo-instruct",
    prompt=prompt,
    max_tokens=500,
    temperature=0.7)

    return response.choices[0].text.strip()

# Function to read file
def read_file(file):
    if file.type == 'application/pdf':
        with pdfplumber.open(BytesIO(file.getvalue())) as pdf:
            return ' '.join(page.extract_text() for page in pdf.pages)
    else:
        return file.getvalue().decode()

# Load CSS
load_css('styles.css')

# Sidebar
st.sidebar.title('Options')

# Add a language selection option
language = st.sidebar.radio('Language', ['English', 'Swedish'])

# Create columns for horizontal layout
col1, col2, col3 = st.sidebar.columns(3)
col4, col5, col6 = st.sidebar.columns(3)

# Organize radio buttons in columns for a horizontal layout
employment_type = col1.radio('Employment Type', ['N/A', 'Full time', 'Part time'])
gender = col2.radio('Gender Preference', ['N/A', 'Male', 'Female', 'Non-binary'])
experience = col3.radio('Experience Preference', ['N/A', 'Entry Level', 'Mid Level', 'Experienced'])
age = col4.radio('Age', ['N/A', 'Young', 'Middle aged', 'Old'])
location = col5.radio('Location', ['N/A', 'On-Site', 'Hybrid', 'Remote'])
driving_license = col6.radio('Driving License', ['N/A', 'Required', 'Not Required'])
education = col1.radio('Education', ['N/A', 'Gymnasial', 'Eftergymnasial/Universitet'])

# Main Area
st.title('CoRecruit AI')

uploaded_file = st.file_uploader("Upload a job posting", type=['txt', 'pdf'])

if uploaded_file is not None:
    # Process the text from the job posting
    text = read_file(uploaded_file)

    # Use the GPT API to recommend changes
    recommendations = get_recommendations(text, gender, experience, age, language)
    st.write(recommendations)

