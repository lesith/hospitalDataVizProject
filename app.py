# app.py
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
from functools import wraps
from dotenv import load_dotenv
import os
import json
from langchain_community.vectorstores import Chroma
from langchain.prompts.chat import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Load environment variables from .env
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Define the folder to save plots
PLOT_FOLDER = 'static/plots'
os.makedirs(PLOT_FOLDER, exist_ok=True)

# Get username and password from environment variables
USERNAME = os.getenv('BASIC_AUTH_USERNAME')
PASSWORD = os.getenv('BASIC_AUTH_PASSWORD')
CHROMA_PATH = os.getenv('CHROMA_PATH')

# Basic authentication decorator
def check_auth(username, password):
    return username == USERNAME and password == PASSWORD

def authenticate():
    return Response(
        'Could not verify your access level for that URL.'
        'You have to login with proper credentials', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# Helper function to save plots as images
def save_plot(filename):
    filepath = os.path.join(PLOT_FOLDER, filename)
    plt.savefig(filepath)
    plt.clf()
    return filepath

@app.route('/')
@requires_auth
def dashboard():
    # Serve the saved images
    plots = {
        'age_distribution': os.path.join(PLOT_FOLDER, 'age_distribution.png'),
        'gender_breakdown': os.path.join(PLOT_FOLDER, 'gender_breakdown.png'),
        'marital_status': os.path.join(PLOT_FOLDER, 'marital_status.png'),
        'encounter_types': os.path.join(PLOT_FOLDER, 'encounter_types.png'),
        'encounter_duration': os.path.join(PLOT_FOLDER, 'encounter_duration.png'),
        'procedure_frequency': os.path.join(PLOT_FOLDER, 'procedure_frequency.png'),
        'encounter_cost_distribution': os.path.join(PLOT_FOLDER, 'encounter_cost_distribution.png'),
        'cost_by_encounter_type': os.path.join(PLOT_FOLDER, 'cost_by_encounter_type.png'),
        'payer_coverage': os.path.join(PLOT_FOLDER, 'payer_coverage.png'),
        'encounters_by_location': os.path.join(PLOT_FOLDER, 'encounters_by_location.png'),
        'reasons_for_encounters': os.path.join(PLOT_FOLDER, 'reasons_for_encounters.png'),
        'procedure_age_groups': os.path.join(PLOT_FOLDER, 'procedure_age_groups.png'),
        'encounters_over_time': os.path.join(PLOT_FOLDER, 'encounters_over_time.png'),
        'peak_hours': os.path.join(PLOT_FOLDER, 'peak_hours.png'),
        'claims_costs_by_payer': os.path.join(PLOT_FOLDER, 'claims_costs_by_payer.png'),
        'out_of_pocket_costs': os.path.join(PLOT_FOLDER, 'out_of_pocket_costs.png')
    }

    return render_template('dashboard.html', plots=plots)


@app.route('/chat', methods=['POST'])
@requires_auth
def chat():
    query_text = request.form['query_text']
    response = chat_query(query_text)
    return jsonify({'query': query_text, 'response': response})


# Chat query function
def chat_query(query_text):
    # Prepare the DB
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB with similarity filtering
    results = db.similarity_search_with_relevance_scores(query_text, k=5)
    print(f"Retrieved results: {results}")

    # Adjusted Similarity Threshold
    if len(results) == 0:
        print("Unable to find matching results.")
        return "Unable to find matching results."

    # Check if any results exceed a set threshold or if the best available should be used
    threshold = 0.7  # change as necessary
    relevant_results = [result for result in results if result[1] >= threshold]

    if len(relevant_results) == 0:
        print("No results exceed the similarity threshold. Returning the best available match.")
        relevant_results = [results[0]]  # Return the best available match if no results exceed threshold

    # Prepare context from the results
    context_texts = []
    document_types = []
    data_sources = []
    field_descriptions_list = []
    patient_ids = []

    for doc, _score in relevant_results:
        context_texts.append(doc.page_content)
        document_types.append(doc.metadata.get("document_type", "unknown"))
        data_sources.append(doc.metadata.get("data_source", "unknown"))
        if "field_descriptions" in doc.metadata:
            field_descriptions_list.append(json.loads(doc.metadata["field_descriptions"]))
        if "PATIENT" in doc.metadata:
            patient_ids.append(doc.metadata["PATIENT"])

    # Compile context with separators
    context_text = "\n\n---\n\n".join(context_texts)
    document_type_context = ", ".join(set(document_types))
    data_source_context = ", ".join(set(data_sources))
    field_descriptions_context = json.dumps(field_descriptions_list, indent=2) if field_descriptions_list else "None"
    patient_ids_context = ", ".join(set(patient_ids)) if patient_ids else "None"

    # Update the prompt template with new metadata
    PROMPT_TEMPLATE = """
        You are an expert data analyst that has access to a hospital dataset that has been denormalized to a single table, 
        and contains encounter information, patient details, payment amount information, insurer details, 
        hospital details and procedures performed on patients.

        Answer user question based on the following context:

        {context}

        ---

        Metadata Information:
        Document Type: {document_type}
        Data Source: {data_source}
        Field Descriptions: {field_descriptions}
        Patient Identifiers: {patient_ids}

        ---

        Answer this question based on the above context: {question}
        """

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        context=context_text,
        question=query_text,
        document_type=document_type_context,
        data_source=data_source_context,
        field_descriptions=field_descriptions_context,
        patient_ids=patient_ids_context
    )

    # Query the model
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    response_text = model.predict(prompt)

    # Compile and return response
    return response_text


# Route for generating each plot
@app.route('/generate/<chart_name>', methods=['POST'])
def generate_chart(chart_name):
    # Whitelist of allowed chart names
    allowed_charts = [
        'age_distribution', 'gender_breakdown', 'marital_status',
        'encounter_types', 'encounter_duration', 'procedure_frequency',
        'encounter_cost_distribution', 'cost_by_encounter_type',
        'payer_coverage', 'encounters_by_location', 'reasons_for_encounters',
        'procedure_age_groups', 'encounters_over_time', 'peak_hours',
        'claims_costs_by_payer', 'out_of_pocket_costs'
    ]

    # Check if the requested chart is allowed
    if chart_name not in allowed_charts:
        return "Chart not found", 404

    # Load hospital_data from SQLite database
    try:
        with sqlite3.connect('healthcare_data.db') as conn:
            merged_df = pd.read_sql_query('SELECT * FROM encounters', conn)
    except Exception as e:
        return f"An error occurred while accessing the database: {str(e)}", 500

    if chart_name == 'age_distribution':
        plt.hist(merged_df['Age'].dropna(), bins=20, edgecolor='black')
        plt.xlabel('Age')
        plt.ylabel('Number of Patients')
        plt.title('Age Distribution of Patients')
        save_plot('age_distribution.png')

    elif chart_name == 'gender_breakdown':
        merged_df['GENDER'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
        plt.title('Gender Breakdown of Patients')
        plt.ylabel('')
        save_plot('gender_breakdown.png')

    elif chart_name == 'marital_status':
        merged_df['MARITAL'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
        plt.xlabel('Marital Status')
        plt.ylabel('Number of Patients')
        plt.title('Marital Status Distribution')
        plt.xticks(rotation=45)
        save_plot('marital_status.png')

    elif chart_name == 'encounter_types':
        plt.figure(figsize=(10, 9))
        merged_df['ENCOUNTERCLASS'].value_counts().plot(kind='bar', color='lightgreen', edgecolor='black')
        plt.xlabel('Encounter Class')
        plt.ylabel('Number of Encounters')
        plt.title('Distribution of Encounter Types')
        plt.xticks(rotation=45)
        save_plot('encounter_types.png')

    elif chart_name == 'encounter_duration':
        avg_duration = merged_df.groupby('ENCOUNTERCLASS')['DURATION_HOURS'].mean()
        plt.bar(avg_duration.index, avg_duration.values, alpha=0.5)
        plt.xlabel('Encounter Class')
        plt.ylabel('Average Duration (Hours)')
        plt.title('Average Encounter Duration by Encounter Class')
        save_plot('encounter_duration.png')

    elif chart_name == 'procedure_frequency':
        plt.figure(figsize=(10, 18))
        merged_df['DESCRIPTION_procedure'].value_counts().head(10).plot(kind='bar', color='lightcoral', edgecolor='black')
        plt.xlabel('Procedure Description')
        plt.ylabel('Number of Procedures')
        plt.title('Top 10 Procedures')
        plt.xticks(rotation=270)
        save_plot('procedure_frequency.png')

    elif chart_name == 'encounter_cost_distribution':
        plt.scatter(range(len(merged_df['TOTAL_CLAIM_COST'].dropna())), merged_df['TOTAL_CLAIM_COST'].dropna(), alpha=0.5)
        plt.xlabel('Index')
        plt.ylabel('Total Claim Cost')
        plt.title('Scatter Plot of Total Claim Costs')
        save_plot('encounter_cost_distribution.png')

    elif chart_name == 'cost_by_encounter_type':
        plt.figure(figsize=(10, 9))
        encounter_cost = merged_df.groupby('ENCOUNTERCLASS')['BASE_ENCOUNTER_COST'].mean()
        encounter_cost.plot(kind='bar', color='mediumseagreen', edgecolor='black')
        plt.xlabel('Encounter Class')
        plt.ylabel('Average Base Cost')
        plt.title('Average Cost by Encounter Type')
        plt.xticks(rotation=45)
        save_plot('cost_by_encounter_type.png')

    elif chart_name == 'payer_coverage':
        plt.figure(figsize=(10, 9))
        payer_coverage = merged_df.groupby('NAME_payer')['PAYER_COVERAGE'].mean().sort_values(ascending=False).head(10)
        payer_coverage.plot(kind='bar', color='mediumpurple', edgecolor='black')
        plt.xlabel('Payer')
        plt.ylabel('Average Coverage')
        plt.title('Top 10 Payers by Average Coverage')
        plt.xticks(rotation=270)
        save_plot('payer_coverage.png')

    elif chart_name == 'encounters_by_location':
        plt.figure(figsize=(10, 9))
        merged_df['CITY'].value_counts().plot(kind='bar', color='gold', edgecolor='black')
        plt.xlabel('City')
        plt.ylabel('Number of Encounters')
        plt.title('Encounters by City')
        plt.xticks(rotation=90)
        save_plot('encounters_by_location.png')

    elif chart_name == 'reasons_for_encounters':
        plt.figure(figsize=(10, 12))
        merged_df['REASONDESCRIPTION'].value_counts().head(10).plot(kind='bar', color='salmon', edgecolor='black')
        plt.xlabel('Reason for Encounter')
        plt.ylabel('Number of Encounters')
        plt.title('Top 10 Reasons for Encounters')
        plt.xticks(rotation=270)
        save_plot('reasons_for_encounters.png')

    elif chart_name == 'procedure_age_groups':
        age_groups = merged_df['Age Group'].dropna().unique()
        descriptions = merged_df['DESCRIPTION_procedure'].dropna().unique()
        for age_group in age_groups:
            for description in descriptions:
                count = merged_df[(merged_df['Age Group'] == age_group) & (merged_df['DESCRIPTION_procedure'] == description)].shape[0]
                if count > 0:
                    plt.scatter([age_group] * count, range(count), alpha=0.5)
        plt.xlabel('Age Group')
        plt.ylabel('Index Range')
        plt.title('Procedure Frequency Across Age Groups (Index Range)')
        plt.xticks(rotation=45)
        save_plot('procedure_age_groups.png')

    elif chart_name == 'encounters_over_time':
        plt.figure(figsize=(10, 9))
        encounters_over_time = merged_df['MONTH'].value_counts().sort_index()
        encounters_over_time.plot(kind='line', marker='o', color='teal')
        plt.xlabel('Month')
        plt.ylabel('Number of Encounters')
        plt.title('Encounters Over Time')
        plt.xticks(rotation=45)
        save_plot('encounters_over_time.png')

    elif chart_name == 'peak_hours':
        merged_df['HOUR'].value_counts().sort_index().plot(kind='bar', color='royalblue', edgecolor='black')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Number of Encounters')
        plt.title('Peak Hours for Encounters')
        plt.xticks(rotation=0)
        save_plot('peak_hours.png')

    elif chart_name == 'claims_costs_by_payer':
        plt.figure(figsize=(10, 12))
        payer_claims_cost = merged_df.groupby('NAME_payer')['TOTAL_CLAIM_COST'].mean().sort_values(ascending=False).head(10)
        payer_claims_cost.plot(kind='bar', color='darkorange', edgecolor='black')
        plt.xlabel('Payer')
        plt.ylabel('Average Claim Cost')
        plt.title('Top 10 Payers by Average Claim Cost')
        plt.xticks(rotation=270)
        save_plot('claims_costs_by_payer.png')

    elif chart_name == 'out_of_pocket_costs':
        avg_out_of_pocket = merged_df.groupby('ENCOUNTERCLASS')['OUT_OF_POCKET'].mean()
        plt.bar(avg_out_of_pocket.index, avg_out_of_pocket.values, alpha=0.5)
        plt.xlabel('Encounter Class')
        plt.ylabel('Average Out-of-Pocket Cost')
        plt.title('Average Out-of-Pocket Cost by Encounter Class')
        save_plot('out_of_pocket_costs.png')

    return redirect(url_for('dashboard'))

if __name__ == "__main__":
    app.run(debug=False)
