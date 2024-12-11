from flask import Flask
from flask import render_template, request , jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import spacy
import re
import PyPDF2
import os

app = Flask(__name__)


model = joblib.load('Model/tfidf_vectorizer.pkl')
nlp = spacy.load('en_core_web_sm')



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


@app.route('/home')
@app.route('/')
def home():
    
    return render_template('home.html')



@app.route('/resume_ranker', methods=['GET', 'POST'])
def resume_ranker():
    if request.method == 'POST':
        # Get job description
        job_description = request.form['job_description']
        job_tokens = processing(job_description)
        job_vector = model.transform([job_tokens]).toarray()

        # Process uploaded resumes
        scores = []
        uploaded_files = request.files.getlist('resumes')
        for resume_file in uploaded_files:
            try:
                file_text = extract_text_from_pdf(resume_file)
                resume_tokens = processing(file_text)
                resume_vector = model.transform([resume_tokens]).toarray()
                similarity = cosine_similarity(resume_vector, job_vector)[0][0]
                scores.append((resume_file.filename, similarity))
            except Exception as e:
                return render_template('resumeranker.html', error=f"Error processing file {resume_file.filename}: {e}")
        # Sort resumes by similarity score
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

        return render_template('results.html', 
                               job_description=job_description, 
                               ranked_resumes=sorted_scores)

    return render_template('resumeranker.html')

@app.route('/ATS', methods=['GET', 'POST'])
def ATS():
    if request.method == 'POST':
        job_description = request.form['job_description']
        job_token = processing(job_description)
        job_vector = model.transform([job_token]).toarray()
        
        uploaded_file = request.files.getlist('resumes')[0]
        try:
            file_text = extract_text_from_pdf(uploaded_file)
            resume_token = processing(file_text)
            resume_vector = model.transform([resume_token]).toarray()
            similarity_score = cosine_similarity(resume_vector, job_vector)[0][0]
            similarity_score = (similarity_score)*100
        except Exception as e:
            return render_template('resumeranker.html', error=f"Error processing file {uploaded_file.filename}: {e}")
        
        return render_template('ATS.html', job_description=job_description, similarity_score = int(similarity_score))
    
    return render_template('ATS.html')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')