from flask import Flask, request, render_template_string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document
import os
from werkzeug.utils import secure_filename

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# HTML template
HTML_TEMPLATE = """
<!doctype html>
<title>Resume Matcher</title>
<h2>Resume to Job Description Matcher</h2>
<form method=post enctype=multipart/form-data>
  Upload Resume (PDF or DOCX): <input type=file name=resume><br><br>
  Paste Job Description:<br><textarea name=jd rows=10 cols=70></textarea><br><br>
  <input type=submit value=Match>
</form>
{% if score is not none %}
  <h3>🔍 Match Score: {{ '{:.2f'.format(score * 100) }}%</h3>
{% endif %}
"""

# DOCX extractor
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

# Resume text extractor (PDF or DOCX)
def extract_resume_text(file_path):
    if file_path.endswith(".pdf"):
        return extract_pdf_text(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        return ""

# Main route
@app.route("/", methods=["GET", "POST"])
def index():
    score = None
    if request.method == "POST":
        jd_text = request.form["jd"]
        resume_file = request.files["resume"]
        filename = secure_filename(resume_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        resume_file.save(filepath)

        resume_text = extract_resume_text(filepath)

        # TF-IDF matching
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([resume_text, jd_text])
        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    return render_template_string(HTML_TEMPLATE, score=score)

if __name__ == "__main__":
    app.run(debug=True)
