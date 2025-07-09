# pip install -r requirements.txt
from flask import Flask, render_template, request
import re
import os
import PyPDF2
import spacy

app = Flask(__name__)

# Load spaCy NLP model for name detection
nlp = spacy.load("en_core_web_sm")

# Define skill keywords
SKILL_KEYWORDS = [
    'python', 'java', 'c++', 'sql', 'javascript', 'react',
    'node', 'html', 'css', 'django', 'flask', 'machine learning',
    'data science', 'aws', 'docker', 'git'
]

# Extract text from uploaded PDF
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Generate AI-style suggestions for missing skills
def generate_suggestions(matched_skills):
    missing_skills = [s for s in SKILL_KEYWORDS if s not in matched_skills]
    if not missing_skills:
        return "Excellent skillset!"
    return f"Consider learning or improving: {', '.join(missing_skills[:5])}"

# def extract_name_from_text(text):
#     # Take the first few lines and try to pick the first proper-looking name
#     lines = text.strip().split("\n")
#     for line in lines[:5]:
#         if len(line.strip().split()) in [2, 3] and line.isupper() is False:
#             return line.strip()
#     return "Not found"

# Analyze resume content
def analyze_resume(text):
    doc = nlp(text)

    # Name detection using spaCy
    name = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), "Not found")

    # Email & phone detection
    email = re.search(r'\b[\w.-]+?@\w+?\.\w+?\b', text)
    phone = re.search(r'\+?\d[\d\s\-\(\)]{7,}\d', text)

    # Skills matching
    matched_skills = [skill for skill in SKILL_KEYWORDS if skill.lower() in text.lower()]
    score = int((len(matched_skills) / len(SKILL_KEYWORDS)) * 100)

    # Suggestions
    suggestions = generate_suggestions(matched_skills)

    return {
        # "name" : name.extract_name_from_text(text) ,
        "email": email.group() if email else "Not found",
        "phone": phone.group() if phone else "Not found",
        "skills": matched_skills,
        "score": score,
        "suggestions": suggestions
    }

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and analysis
@app.route('/upload', methods=['POST'])
def upload():
    if 'resume' not in request.files:
        return "No file uploaded"
    
    file = request.files['resume']
    if file.filename == '':
        return "No selected file"

    # Save file temporarily
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    # Extract and analyze text
    text = extract_text_from_pdf(filepath)
    data = analyze_resume(text)

    # Cleanup
    os.remove(filepath)

    return render_template('result.html', data=data)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
