import os
import PyPDF2
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools import ArxivQueryRun
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from litellm import completion
from langchain.embeddings import HuggingFaceEmbeddings


# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
'''
if huggingface_token:
    login(token=huggingface_token)
'''



# Initialize models and tools
text_embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

arxiv_tool = ArxivQueryRun()


def extract_text_from_pdfs(uploaded_files):
    all_text = ""
    for uploaded_file in uploaded_files:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            all_text += page.extract_text() or ""
    return all_text

def process_text_and_store(all_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(all_text)

    # Create FAISS index
    faiss_db = FAISS.from_texts(chunks, embedding_model)
    return faiss_db

def semantic_search(query, faiss_db, top_k=2):
    docs = faiss_db.similarity_search(query, k=top_k)
    return docs

def generate_response(query, context):
    prompt = f"Query: {query}\nContext: {context}\nAnswer:"
    response = completion(
        model="gemini/gemini-1.5-flash",
        messages=[{"content": prompt, "role": "user"}],
        api_key=gemini_api_key
    )
    return response['choices'][0]['message']['content']

def main():
    st.title("RAG-powered Research Paper Assistant 📚✨")

    option = st.radio("Choose an option:", ("Upload PDFs", "Search arXiv"))

    if option == "Upload PDFs":
        uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
        if uploaded_files:
            st.write("Processing uploaded files...")
            all_text = extract_text_from_pdfs(uploaded_files)
            faiss_db = process_text_and_store(all_text)
            st.session_state['faiss_db'] = faiss_db
            st.success("PDF content processed and stored successfully!")

            query = st.text_input("Enter your query about the PDFs:")
            if st.button("Execute Query") and query:
                docs = semantic_search(query, st.session_state['faiss_db'])
                context = "\n".join([doc.page_content for doc in docs])
                response = generate_response(query, context)
                st.subheader("Generated Response:")
                st.write(response)

    elif option == "Search arXiv":
        search_query = st.text_input("Enter your search query for arXiv:")

        if st.button("Search ArXiv") and search_query:
            arxiv_results = arxiv_tool.invoke(search_query)
            st.session_state["arxiv_results"] = arxiv_results

            st.subheader("Search Results:")
            st.write(arxiv_results)

            faiss_db = process_text_and_store(arxiv_results)
            st.session_state["faiss_db"] = faiss_db

            st.success("arXiv paper content processed and stored successfully!")

        if "arxiv_results" in st.session_state and "faiss_db" in st.session_state:
            query = st.text_input("Ask a question about the paper:")
            if st.button("Execute Query on Paper") and query:
                docs = semantic_search(query, st.session_state["faiss_db"])
                context = "\n".join([doc.page_content for doc in docs])
                response = generate_response(query, context)
                st.subheader("Generated Response:")
                st.write(response)

if __name__ == "__main__":
    main()
