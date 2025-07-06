from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_match_score(resume_text: str, jd_text: str) -> float:
    """
    Calculate similarity score between resume and job description.
    Returns a float between 0.0 and 1.0
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0  # Handle empty input

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return score

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate resume-JD match score.")
    parser.add_argument("--resume", type=str, help="Path to resume .txt file", required=True)
    parser.add_argument("--jd", type=str, help="Path to JD .txt file", required=True)
    args = parser.parse_args()

    with open(args.resume, "r", encoding="utf-8") as f1, open(args.jd, "r", encoding="utf-8") as f2:
        resume = f1.read()
        jd = f2.read()

    score = calculate_match_score(resume, jd)
    print(f"Match Score: {score * 100:.2f}%")
