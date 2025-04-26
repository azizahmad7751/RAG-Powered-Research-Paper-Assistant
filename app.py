import os
import PyPDF2
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools import ArxivQueryRun
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from litellm import completion

# -------------------------
# Define wrapper class here
class MyHuggingFaceEmbeddings:
    def __init__(self, api_key, model_name):
        self.client = HuggingFaceInferenceAPIEmbeddings(
            api_key=api_key,
            model_name=model_name
        )
    
    def embed_documents(self, texts):
        return self.client.embed_documents(texts)

    def embed_query(self, text):
        return self.client.embed_query(text)
# -------------------------

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

# Initialize models
embedding_model = MyHuggingFaceEmbeddings(
    api_key=huggingface_token,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

arxiv_tool = ArxivQueryRun()


import requests

def embed_texts(texts, hf_token):
    headers = {
        "Authorization": f"Bearer {hf_token}"
    }
    API_URL = "https://api-inference.huggingface.co/embeddings/sentence-transformers/all-MiniLM-L6-v2"
    
    payload = {"inputs": texts}
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"Embedding failed: {response.text}")
    
    embeddings = response.json()
    return embeddings


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

    # Embed texts manually
    embeddings = embed_texts(chunks, huggingface_token)

    # Create FAISS index
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    # Store texts alongside
    faiss_db = FAISS(embedding_function=None, index=index, documents=chunks)

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
