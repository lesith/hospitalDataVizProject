# Seneca Fall 2024 - Business Case and AI Competition

## Setting up the Flask Web App

### Requirements:
* Docker
* Docker Compose plugin
* Pre-generated Chrome DB files in the ``chroma_db`` folder in project root
* API key from Google AI Studio

### Steps:
1. Clone the git repo
2. Create ``.env`` using the ``env.sample`` within the project root
3. Insert the API Key in ``.env``
3. Build command: ``docker compose build``
4. Run command: ``docker compose up -d``
5. Access via ``http://0.0.0.0:8000``

## Generating the Chroma DB

### Requirements:
* Google Colab access or a local Jupyter server

### Steps:
1. Follow through each step of the ``mergedHospitalDataRAG.ipynb`` notebook until ``save_to_chroma(chunks)`` (including)

