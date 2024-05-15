import os
import streamlit as st
from openai import OpenAI
import pdfplumber
from io import BytesIO

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

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

def get_recommendations(text, employment_type, gender, experience, age, location, driving_license, education, language):
    if language == 'Swedish':
        prompt = (f"{text}\n\nGivet att den ideala kandidaten är {employment_type}, {gender}, {experience}, {age}, "
                  f"{location}, {driving_license} och {education}, hur kan denna jobbannons förbättras?")
        system_message = "Du är en hjälpsam assistent."
    else:  # Default to English
        prompt = (f"{text}\n\nGiven that the ideal candidate is {employment_type}, {gender}, {experience}, {age}, "
                  f"{location}, {driving_license} and {education}, how could this job posting be improved?")
        system_message = "You are a helpful assistant."

    try:
        response = client.completions.create(
            model="ft:gpt-3.5-turbo-0125:personal::9N4jESmA",
            prompt=prompt,
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Function to read file
def read_file(file):
    if file.type == 'application/pdf':
        with pdfplumber.open(BytesIO(file.getvalue())) as pdf:
            return ' '.join(page.extract_text() for page in pdf.pages)
    else:
        return file.getvalue().decode()

# Sidebar
st.sidebar.title('Options')

# Add a language selection option
language = st.sidebar.radio('Language', ['English', 'Swedish'])

employment_type = st.sidebar.radio('Employment Type', ['N/A', 'Full time', 'Part time'])
gender = st.sidebar.radio('Gender Preference', ['N/A', 'Male', 'Female', 'Non-binary'])
experience = st.sidebar.radio('Experience Preference', ['N/A', 'Entry Level', 'Mid Level', 'Experienced'])
age = st.sidebar.radio('Age', ['N/A', 'Young', 'Middle aged', 'Old'])
location = st.sidebar.radio('Location', ['N/A', 'On-Site', 'Hybrid', 'Remote'])
driving_license = st.sidebar.radio('Driving License', ['N/A', 'Required', 'Not Required'])
education = st.sidebar.radio('Education', ['N/A', 'Gymnasial', 'Eftergymnasial/Universitet'])

# Main Area
st.title('CoRecruit AI')

uploaded_file = st.file_uploader("Upload a job posting", type=['txt', 'pdf'])

if uploaded_file is not None:
    # Process the text from the job posting
    text = read_file(uploaded_file)

    # Use the GPT API to recommend changes
    recommendations = get_recommendations(text, employment_type, gender, experience, age, location, driving_license, education, language)
    
    if recommendations:
        st.write(recommendations)
