from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()

# ---- STEP 1: Load ALL PDFs ----
pdf_folder = r"C:\Users\nirun\Desktop\rag project!\data\visualization"

print("Loading PDFs...")
all_pages = []
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        filepath = os.path.join(pdf_folder, filename)
        loader = PyPDFLoader(filepath)
        pages = loader.load()
        all_pages.extend(pages)
print(f"Total pages loaded: {len(all_pages)} ✅")

# ---- STEP 2: Split into chunks ----
print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(all_pages)
print(f"Total chunks created: {len(chunks)} ✅")

# ---- STEP 3: Load Local Embeddings ----
print("Loading local embeddings model...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
print("Embeddings model loaded! ✅")

# ---- STEP 4: Create or Load Vector Database ----
if os.path.exists("vector_db"):
    print("Loading existing vector database...")
    vector_store = FAISS.load_local(
        "vector_db",
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("Vector database loaded! ✅")
else:
    print("Creating new vector database...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("vector_db")
    print("Vector database created and saved! ✅")

# ---- STEP 5: Connect Gemini AI ----
print("Connecting to Gemini AI...")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3
)

# ---- STEP 6: Create Q&A Chain ----
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based only on the context below.
If the answer is not in the context, say "I could not find the answer in the provided documents."

Context: {context}

Question: {question}
""")

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("Q&A chain ready! ✅")

# ---- STEP 7: Test Question ----
question = "What is data visualization?"
print(f"\nQuestion: {question}")
answer = chain.invoke(question)
print(f"\nAnswer: {answer}")