import streamlit as st
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

# ---- Page Config ----
st.set_page_config(
    page_title="MBA Notes Q&A",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 MBA Notes Q&A System")
st.markdown("Ask any question from your **Data Visualization notes!**")
st.divider()

# ---- Load Everything ----
@st.cache_resource
def load_qa_chain():
    # Load local embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Load vector database
    vector_store = FAISS.load_local(
        "vector_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Connect Gemini AI
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3
    )

    # Create prompt
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Answer the question based only on the context below.
    If the answer is not in the context, say "I could not find the answer in the provided documents."

    Context: {context}

    Question: {question}
    """)

    # Create chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# Load the chain
with st.spinner("Loading your MBA notes... Please wait!"):
    chain = load_qa_chain()

st.success("Ready! Ask me anything from your notes! ✅")
st.divider()

# ---- Question Input ----
question = st.text_input(
    "💬 Type your question here:",
    placeholder="e.g. What is data visualization?"
)

if st.button("Get Answer 🚀"):
    if question:
        with st.spinner("Searching your notes..."):
            answer = chain.invoke(question)
        st.markdown("### 📝 Answer:")
        st.write(answer)
    else:
        st.warning("Please type a question first!")