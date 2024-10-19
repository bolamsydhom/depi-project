
import time
import psutil
import glob
import PyPDF2
#import chromadb
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer, AutoModelForCausalLM,
    pipeline
)
from transformers import LlamaTokenizer, LlamaForCausalLM,BitsAndBytesConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_cpp import Llama



def RAG_Chain(pdf_file,question):
    model_path = "/home/mona/Downloads/Pubmed_model_GGUF"
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    doc = ""
    for page_num in range(len(pdf_reader.pages) ):
        page = pdf_reader.pages[page_num] 
        doc += page.extract_text()

    # Check if any documents were loaded
    if not doc:
        raise ValueError("No documents found. Please check the PDF directory path.")
    
    # Split the loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(doc)

    # Create HuggingFace embeddings and vector store
    embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # Efficient model suitable for most tasks
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    import chromadb

    chromadb.api.client.SharedSystemClient.clear_system_cache()
   
    vectorstore = Chroma.from_texts(texts=splits, embedding=embeddings)

        # Define the retriever using Chroma
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(question)
    if not retrieved_docs:
        return "No relevant information found in the documents."
    
    # Format the context
    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # Prepare the prompt for the LLM
    formatted_prompt = (
        f"Answer the question based on the context below.\n\n"
        f"Context:\n{formatted_context}\n\nQuestion: {question}\n\nAnswer:"
    )

     # Instantiate the Llama model using the gguf file

    llama_model = Llama(
        model_path,
        n_ctx=2048,          # Context length
        n_threads=8,         # Number of CPU threads to use
        temperature=0.7,      # Sampling temperature
        n_gpu_layers=-1
    )

    # Generate the answer
    answer = llama_model(formatted_prompt)
    return answer["choices"][0]["text"]

