# 🤖 RAG Q&A System

A complete Retrieval-Augmented Generation (RAG) 
system that answers questions from your own PDF documents!

## 🚀 Tech Stack
- 🐍 Python
- 🦜 LangChain
- 🗄️ FAISS Vector Database
- 🤗 HuggingFace Embeddings
- 💎 Google Gemini AI
- 🖥️ Streamlit UI
- 🐳 Docker

## ⚙️ Setup

### 1. Clone the repository
git clone https://github.com/yourusername/rag-qa-system.git

### 2. Create .env file
GOOGLE_API_KEY=your_gemini_api_key_here

### 3. Add your PDFs
Add PDF files to data/ folder

### 4. Run with Docker
docker build -t rag-qa-system .
docker run -p 8501:8501 --env-file .env rag-qa-system

### 5. Open browser
http://localhost:8501

## 📚 How RAG works
1. PDFs loaded and split into chunks
2. Chunks converted to embeddings (HuggingFace)
3. Stored in FAISS vector database
4. Your question searched against chunks
5. Top 3 relevant chunks sent to Gemini AI
6. Gemini generates smart answer!

## 🎓 Built by Nikhil Chandra
