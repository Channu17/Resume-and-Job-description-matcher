from fastapi import FastAPI, File, UploadFile
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import spacy
import re
import PyPDF2
import io

# Load pre-trained TF-IDF vectorizer and spaCy model
model = joblib.load('F:/Model/Model/tfidf_vectorizer.pkl')
nlp = spacy.load('en_core_web_sm')

app = FastAPI()

def processing(content):
    """Process the text content: tokenize, lemmatize, and clean."""
    doc = nlp(content)
    processed_tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    processed_text = ' '.join(processed_tokens)
    processed_text = re.sub(r'[^A-Za-z0-9\s]', '', processed_text)
    return processed_text.lower()

def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@app.post("/match-resumes/")
async def match_resumes(
    job_description: str,
    resumes: List[UploadFile] = File(...)
):
    # Process job description
    job_tokens = processing(job_description)
    job_vector = model.transform([job_tokens]).toarray()

    # Process resumes and calculate similarity
    scores = []
    for idx, resume_file in enumerate(resumes):
        file_content = await resume_file.read()
        file_text = extract_text_from_pdf(io.BytesIO(file_content))
        resume_tokens = processing(file_text)
        resume_vector = model.transform([resume_tokens]).toarray()
        similarity = cosine_similarity(resume_vector, job_vector)[0][0]
        scores.append((resume_file.filename, similarity))

    # Sort resumes by similarity score in ascending order
    sorted_scores = sorted(scores, key=lambda x: x[1])

    return {
        "job_description": job_description,
        "ranked_resumes": [
            {"filename": filename, "similarity_score": score}
            for filename, score in sorted_scores
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
