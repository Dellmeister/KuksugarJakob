import os
import streamlit as st
import openai
import pdfplumber
from io import BytesIO
from pdf2image import convert_from_bytes
import pytesseract

# Initialize OpenAI client with the API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

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

def get_recommendations(text, gender, experience, age, language, employment_type, location, driving_license, education):
    if language == 'Swedish':
        prompt = f"{text}\n\nJag har en jobbannons och jag vill förbättra den baserat på vissa kriterier. Den ideala kandidaten för min jobbannons har följande egenskaper: {employment_type}, {gender}, {experience}, {age}, {location}, {driving_license} och {education}. Kan du ge en översiktlig bedömning av jobbannonsen och kommentera specifika meningar, ord eller stycken som kan förbättras eller ändras för att bättre attrahera den ideala kandidaten? Skriv svaret på Svenska."
        system_message = "Du är en hjälpsam assistent."
    else:  # Default to English
        prompt = f"{text}\n\nGiven that the ideal candidate is {gender}, {experience}, and {age}, how could this job posting be improved?"
        system_message = "You are a helpful assistant."

    response = openai.ChatCompletion.create(
        model="ft:gpt-3.5-turbo-0125:personal::9N4jESmA",  # Use your fine-tuned chat model
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )
    return response["choices"][0]["message"]["content"]

# Function to read file with OCR fallback
def read_file(file):
    if file.type == 'application/pdf':
        try:
            with pdfplumber.open(BytesIO(file.getvalue())) as pdf:
                text = ' '.join(page.extract_text() for page in pdf.pages if page.extract_text())
                if not text.strip():
                    raise ValueError("No text found in PDF pages.")
                return text
        except Exception as e:
            st.warning(f"Standard PDF text extraction failed: {e}. Trying OCR...")
            try:
                images = convert_from_bytes(file.getvalue())
                text = ' '.join(pytesseract.image_to_string(image) for image in images)
                if not text.strip():
                    raise ValueError("No text found in OCR processed images.")
                return text
            except Exception as ocr_e:
                st.error(f"OCR text extraction failed: {ocr_e}")
                return ""
    else:
        try:
            return file.getvalue().decode()
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return ""

# Load CSS
load_css('styles.css')

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
    
    # Display the extracted text to verify
    st.subheader("Extracted Text from the Uploaded File:")
    st.write(text)
    
    # Use the GPT API to recommend changes if text extraction is successful
    if text:
        recommendations = get_recommendations(text, gender, experience, age, language, employment_type, location, driving_license, education)
        st.subheader("Recommendations:")
        st.write(recommendations)
    else:
        st.error("Failed to extract text from the uploaded file.")
