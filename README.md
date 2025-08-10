# RAG-Medical-Bot

# How to run?
### STEPS:

Clone the repository

```bash
git clone https://github.com/gt951436/RAG-Medical-Bot.git
```

### STEP 01 - Create a conda environment after opening the repository

```bash
conda create -n medbot python=3.10 -y
```

```bash
conda activate medbot
```


### STEP 02- install the requirements

```bash
pip install -r requirements.txt
```

### Create a .env file in the root directory and add your Pinecone & openai credentials as follows:
```bash
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GOOGLE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```
```bash
# run the following command to store embeddings to pinecone
python store_index.py
```
```bash
# Finally run the following command
python app.py
```

### Now,
```bash
open localhost:
```

### Techstack used
- Python
- Langchain
- Flask
- GPT (Gemini API key, because OpenAPI api key was notworking for me)
- Pinecone
- Docker



