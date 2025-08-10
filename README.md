# RAG-Medical-Bot

# How to run?
### STEPS:

Clone the repository

```bash
git clone https://github.com/gt951436/RAG-Medical-Bot.git
cd Rag-Medical-Bot
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

### 🐳 Containerization with Docker

#### 🚀 Build and Run

### 1. Build the Docker image
```bash
docker-compose build
```

### 2. Start the containers
```bash
docker-compose up
```

#### This will:
- Build the app image
- Start the RAG bot backend service
- Expose it on port 8080 (or whichever you configured in docker-compose.yml)

### 3. Access the app
- API endpoint: http://localhost:8080

### 🛠 Development mode
If you want to run the container in development mode (with hot reload for code changes):
```bash
docker-compose up --build
```

### 🧹 Stopping & Cleaning
#### To stop the containers:
```bash
docker-compose down
```

### 📂 .dockerignore
We use a .dockerignore file to avoid copying unnecessary files into the image.

```bash
# .dockerignore file includes:

__pycache__/
*.pyc
*.pyo
*.pyd
.Python
.env
.git
.gitignore
.vscode
*.db
*.sqlite3
*.log
.DS_Store
*.egg-info
build/
dist/

```



