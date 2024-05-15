import openai
import logging

# Initialize OpenAI client with the API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

def get_recommendations(text, gender, experience, age, language, employment_type, location, driving_license, education):
    if language == 'Swedish':
        prompt = f"{text}\n\nGivet att den ideala kandidaten är {employment_type}, {gender}, {experience}, {age}, {location}, {driving_license} och {education}, hur kan denna jobbannons förbättras?"
        system_message = "Du är en hjälpsam assistent."
    else:  # Default to English
        prompt = f"{text}\n\nGiven that the ideal candidate is {employment_type}, {gender}, {experience}, {age}, {location}, {driving_license}, and {education}, how could this job posting be improved?"
        system_message = "You are a helpful assistant."

    response = openai.ChatCompletion.create(
        model="ft:gpt-3.5-turbo-0125:personal::9N4jESmA",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )

    return response.choices[0].message['content'].strip()

# Example usage
if __name__ == "__main__":
    # Replace with your actual values
    text = "Can you provide a recommendation based on my profile?"
    gender = "male"
    experience = "5 years"
    age = "30"
    language = "English"
    employment_type = "Full time"
    location = "Remote"
    driving_license = "Required"
    education = "University"

    recommendations = get_recommendations(text, gender, experience, age, language, employment_type, location, driving_license, education)
    print(recommendations)
