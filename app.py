import os
from dotenv import load_dotenv
import gradio as gr
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.chains import RetrievalQA

load_dotenv(dotenv_path="config.env")

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please set it in config.env")

MODEL_NAME = "gpt-3.5-turbo"
llm = ChatOpenAI(model=MODEL_NAME, temperature=0.3, openai_api_key=openai_api_key)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

UPLOAD_FOLDER = "uploads"
INDEX_FOLDER = "storage"
INDEX_FILE = os.path.join(INDEX_FOLDER, "my_faiss_index.index")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)

vector_store = None

def create_empty_faiss_index(dim=384):
    """Create an empty FAISS index with synchronized docstore."""
    index = faiss.IndexFlatL2(dim) 
    docstore = InMemoryDocstore({})
    index_to_docstore_id = {}
    return FAISS(embedding_function=embedding_model.embed_query, 
                 index=index, 
                 docstore=docstore, 
                 index_to_docstore_id=index_to_docstore_id)

def load_index():
    """Load the FAISS index or create a new one if it doesn't exist."""
    global vector_store
    if os.path.exists(INDEX_FILE):
        try:
            print("Loading FAISS index from disk...")
            index = faiss.read_index(INDEX_FILE)
            docstore = InMemoryDocstore({})
            index_to_docstore_id = {}
            vector_store = FAISS(embedding_model.embed_query, index, docstore, index_to_docstore_id)
            
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            print("Deleting corrupted index and starting fresh.")
            os.remove(INDEX_FILE)
            vector_store = create_empty_faiss_index()
    else:
        print("No existing FAISS index found. Starting fresh.")
        vector_store = create_empty_faiss_index()

def save_index():
    """Save the current FAISS index to disk."""
    if vector_store and vector_store.index:
        faiss.write_index(vector_store.index, INDEX_FILE)
        print("Saved FAISS index to disk...")
    else:
        print("No index to save.")

def process_pdfs(pdf_files):
    """Process uploaded PDFs and add their contents to the FAISS index."""
    global vector_store
    all_docs = []

    if not pdf_files:
        return "No PDF files provided. Please upload valid files."

    if isinstance(pdf_files, str):
        pdf_files = [pdf_files]

    for pdf_path in pdf_files:
        if not os.path.isfile(pdf_path):
            return f"Error processing {pdf_path}: File does not exist."

        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_docs = splitter.split_documents(documents)
            all_docs.extend(split_docs)
        except Exception as e:
            return f"Error processing {pdf_path}: {str(e)}"

    if all_docs:
        vector_store.add_documents(all_docs)
        save_index()
        return f"Processed {len(all_docs)} documents and updated the FAISS index."
    else:
        return "No valid text found in the uploaded PDFs."

def answer_question(query):
    """Answer questions based on the FAISS index and GPT-3.5-turbo."""
    print(f"DEBUG: FAISS index size: {vector_store.index.ntotal}")
    print("DEBUG: Document store keys:", list(vector_store.docstore._dict.keys()))
    print("DEBUG: Index-to-Docstore ID mapping:", vector_store.index_to_docstore_id)
    if vector_store is None or vector_store.index.ntotal == 0:
        return "No documents found. Please upload and process PDFs first."

    print(f"DEBUG: FAISS index size: {vector_store.index.ntotal}")
    if vector_store.index.ntotal == 0:
        return "No documents found in the FAISS index. Please upload and process PDFs first."

    try:
        query_embedding = embedding_model.embed_query(query)
        print("DEBUG: Query embedding generated successfully. Length:", len(query_embedding))
    except Exception as e:
        print(f"DEBUG: Error generating query embedding: {e}")
        return "Error generating query embedding. Please check embedding model."

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    try:
        documents = retriever.get_relevant_documents(query)
        print("DEBUG: Retrieved documents:")
        if not documents:
            return "No relevant documents found for the query."
        for i, doc in enumerate(documents):
            print(f"Document {i+1}: {doc.page_content[:100]}...")
    except Exception as e:
        print(f"DEBUG: Error retrieving documents: {e}")
        return "Error retrieving relevant documents."

    context = "\n".join([doc.page_content for doc in documents])
    prompt = f"""
    Use the following context to answer the question concisely and accurately:

    Context:
    {context}

    Question: {query}
    Answer:
    """
    try:
        response = llm.invoke(prompt)
        return response.content.strip() if hasattr(response, "content") else "Unexpected GPT-3.5-turbo response format."
    except Exception as e:
        return f"Error generating answer with GPT-3.5-turbo: {str(e)}"




with gr.Blocks() as demo:
    gr.Markdown("# PDF-based Chatbot with FAISS and GPT-3.5-turbo")

    with gr.Row():
        pdf_input = gr.File(label="Upload PDF Files", file_types=[".pdf"], interactive=True)
        upload_button = gr.Button("Process PDFs")

    upload_output = gr.Textbox(label="Upload Status")

    question_input = gr.Textbox(label="Ask a Question")
    question_button = gr.Button("Get Answer")
    answer_output = gr.Textbox(label="Answer")

    upload_button.click(process_pdfs, inputs=pdf_input, outputs=upload_output)
    question_button.click(answer_question, inputs=question_input, outputs=answer_output)

if __name__ == "__main__":
    load_index()
    demo.launch()
