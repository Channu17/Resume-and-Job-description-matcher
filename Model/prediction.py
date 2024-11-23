import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re


model  = joblib.load('tfidf_vectorizer.pkl')

job_input = ['''
Manage and optimize content creation, including blogs, newsletters, video content, and social media posts.
Collaborate with product and sales teams to align marketing initiatives with product launches and sales targets.
Conduct market research to identify consumer trends, competitor activities, and new opportunities.
Track and analyze key marketing metrics and report on campaign performance.
Manage the marketing budget, ensuring efficient allocation of resources across various initiatives.
Lead, mentor, and manage the marketing team, providing direction and support for their professional development.
Ensure consistency in brand messaging and maintain the company's voice across all marketing materials.
Stay up-to-date with the latest marketing trends, tools, and technologies to ensure the company remains competitive.
Requirements:
Bachelor's degree in Marketing, Business, Communications, or a related field (or equivalent experience).
3+ years of experience in marketing, with at least 1-2 years in a managerial or leadership role.
Strong experience in digital marketing, including SEO/SEM, email marketing, social media, and content marketing.
Proven ability to develop and execute marketing strategies that align with business objectives.
Exceptional written and verbal communication skills.
Strong analytical skills and experience with data-driven decision-making.
Proficient in marketing tools such as Google Analytics, HubSpot, Mailchimp, SEMrush, and social media management platforms.
Experience in managing a marketing team and working collaboratively with cross-functional teams.
Ability to manage multiple projects simultaneously and meet deadlines in a fast-paced environment.
Preferred Skills:
Master’s degree in Marketing or Business.
Experience with paid search (Google Ads, Facebook Ads, etc.).
Knowledge of graphic design software (e.g., Adobe Creative Suite).
Familiarity with marketing automation platforms.
Experience in B2B or B2C marketing, depending on the company’s focus.
Benefits:
Competitive salary and bonus structure.
Health, dental, and vision insurance.
Paid time off and holidays.
Retirement plan with company match.
Opportunities for professional growth and development.
Flexible work schedule and remote work options.

''']

resume_input = ['''Name: John Doe
Contact: johndoe@email.com | (123) 456-7890
LinkedIn: linkedin.com/in/johndoe

Objective: A data-driven professional with 3+ years of experience in analyzing data and providing actionable insights. Skilled in SQL, Python, Tableau, and data visualization. Looking for a challenging Data Analyst role to contribute to data-driven decision-making.

Skills:

SQL (MySQL, PostgreSQL)
Python (Pandas, NumPy)
Tableau
Data Analysis & Visualization
Statistical Analysis
Data Cleaning & Preprocessing
Experience:

Data Analyst | XYZ Corp.
March 2021 - Present

Developed and maintained interactive dashboards using Tableau.
Wrote SQL queries to extract, filter, and aggregate large datasets.
Analyzed sales data to identify trends and forecast future sales.
Collaborated with marketing and finance teams to improve business processes.
Junior Data Analyst | ABC Ltd.
June 2019 - February 2021

Assisted senior analysts in data collection, cleaning, and preprocessing.
Conducted statistical analysis to measure customer satisfaction.
Automated daily reporting tasks using Python.
Education:

Bachelor of Science in Computer Science
University of XYZ, Graduated: May 2019''',
'''Name: Jane Smith
Contact: janesmith@email.com | (987) 654-3210
LinkedIn: linkedin.com/in/janesmith

Objective: A skilled software engineer with 4+ years of experience in developing scalable and efficient software solutions using Java, Spring Framework, and microservices. Looking for a challenging role to leverage my expertise in full-stack development and cloud technologies.

Skills:

Java (Spring, Hibernate)
RESTful APIs
Microservices Architecture
AWS, Docker, Kubernetes
ReactJS
Git, CI/CD
Experience:

Software Engineer | Tech Solutions Inc.
January 2020 - Present

Designed and implemented a microservices-based architecture for a cloud-native application using Spring Boot.
Developed RESTful APIs for seamless integration with front-end applications.
Worked with AWS services like S3, Lambda, and DynamoDB.
Collaborated with front-end developers to build an intuitive user interface with React.
Junior Software Developer | CodeLab
July 2018 - December 2019

Assisted in the development of Java-based applications using the Spring Framework.
Wrote unit and integration tests to ensure application reliability.
Participated in daily Agile sprints and sprint retrospectives.
Education:

Bachelor of Technology in Computer Science
ABC University, Graduated: May 2018
''',
'''Name: Sarah Johnson
Contact: sarahjohnson@email.com | (555) 123-4567
LinkedIn: linkedin.com/in/sarahjohnson

Objective: Dynamic and results-driven Marketing Manager with 5+ years of experience in digital marketing, campaign management, and brand development. Passionate about leveraging data-driven strategies to drive growth and engagement. Looking for a challenging position to utilize my skills and expertise in a forward-thinking company.

Skills:

Digital Marketing (SEO, SEM, Social Media)
Brand Strategy
Marketing Analytics (Google Analytics, HubSpot)
Campaign Management
Team Leadership and Collaboration
Content Creation and Management
Experience:

Marketing Manager | ABC Marketing Agency
January 2020 - Present

Developed and executed successful digital marketing campaigns across SEO, Google Ads, and social media platforms.
Led a team of marketing professionals to increase brand awareness and customer engagement.
Analyzed campaign performance and optimized marketing strategies to improve ROI.
Managed content creation for email campaigns, blogs, and social media posts.
Digital Marketing Specialist | XYZ Corporation
June 2017 - December 2019

Managed and optimized paid search campaigns (Google Ads, Bing Ads).
Conducted A/B testing on landing pages to increase conversion rates.
Worked closely with product and sales teams to align marketing strategies with business goals.
Education:

Bachelor of Science in Marketing
University of ABC, Graduated: May 2017''',
'''John Doe
john.doe@email.com | (123) 456-7890 | LinkedIn: linkedin.com/in/johndoe | GitHub: github.com/johndoe

Objective:
Aspiring Web Developer with a solid foundation in front-end and back-end development. Eager to apply skills in HTML, CSS, JavaScript, and React to contribute to a growing web development team.

Skills:

HTML, CSS, JavaScript
React.js, Bootstrap, jQuery
Git, GitHub, Version Control
Basic knowledge of Node.js
Responsive Web Design
RESTful APIs
Experience:
Intern | Web Development Intern
Tech Solutions Inc. | June 2023 – August 2023

Assisted in building responsive websites using HTML, CSS, and JavaScript.
Worked with the design team to convert mockups into fully functional web pages.
Optimized web applications for mobile devices and various screen sizes.
Wrote clean and reusable code and participated in daily stand-ups and code reviews.
Education:
Bachelor of Science in Computer Science
University of California, Los Angeles | Graduated: May 2023

Relevant coursework: Web Development, JavaScript Programming, Data Structures'''
]

nlp = spacy.load('en_core_web_sm')
def processing(content):
    doc = nlp(content)
    processed_tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    processed_text = ' '.join(processed_tokens)
    processed_text= re.sub(r'[^A-Za-z0-9\s]', '', processed_text)
    return processed_text.lower()


job_tokens= [processing(job) for job in job_input]
resume_tokens = [processing(resume) for resume in resume_input]

resume_vectors = model.transform(resume_tokens).toarray()
job_vectors = model.transform(job_tokens).toarray()

sim= []
job_vector = job_vectors[0].reshape(1, -1)
for idx,resume_vector in enumerate(resume_vectors):
    resume_vector = resume_vector.reshape(1, -1)
    similarity = cosine_similarity(resume_vector, job_vector)
    sim.append((idx, similarity))
    
sorted_sim = sorted(sim, key=lambda x: x[1], reverse=True)

for idx, score in sorted_sim:
    print(f"Resume index: {idx+1}, Similarity score: {score}")

