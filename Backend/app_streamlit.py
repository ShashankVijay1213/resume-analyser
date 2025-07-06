import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document
import tempfile

# Extract DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Unified extractor
def extract_resume_text(uploaded_file):
    suffix = uploaded_file.name.lower()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if suffix.endswith(".pdf"):
        return extract_pdf_text(tmp_path)
    elif suffix.endswith(".docx"):
        return extract_text_from_docx(tmp_path)
    else:
        return ""

# Streamlit UI
st.title("🧠 Resume vs JD Matcher")
st.markdown("Upload your **resume** and paste a **job description** to see the match score.")

resume_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
jd_text = st.text_area("Paste Job Description", height=200)

if resume_file and jd_text:
    resume_text = extract_resume_text(resume_file)

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    st.success(f"✅ Match Score: **{score * 100:.2f}%**")

    # Optional Keyword Suggestions
    resume_words = set(resume_text.lower().split())
    jd_words = set(jd_text.lower().split())
    missing_keywords = jd_words - resume_words
    st.markdown("🔍 **Missing Keywords in Resume:**")
    st.code(", ".join(sorted(missing_keywords)))

from scorer import calculate_match_score

score = calculate_match_score(resume_text, jd_text)
