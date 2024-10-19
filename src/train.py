import gradio as gr
import time
import psutil
import glob
import mlflow
import mlflow.pytorch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
import os
# Specify tracking server
mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5003')
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Load PDFs from a directory
pdf_directory = './Application'  # Update this path to your PDFs
pdf_files = glob.glob(f'{pdf_directory}/*.pdf')

docs = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    docs.extend(loader.load())

if not docs:
    raise ValueError("No documents found. Please check the PDF directory path.")

# Split the loaded documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Create HuggingFace embeddings and vector store
embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Load the LLM model
llm_model_name = 'PubMed_Llama3.1_Based_model'
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
model = AutoModelForCausalLM.from_pretrained(llm_model_name)

# Create the HuggingFace pipeline
hf_pipeline = pipeline(
    'text2text-generation',
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    do_sample=False
)

# Wrap the pipeline in a LangChain LLM
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Define the retriever using Chroma
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

def rag_chain(question):
    retrieved_docs = retriever.get_relevant_documents(question)
    if not retrieved_docs:
        return "No relevant information found in the documents."
    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    formatted_prompt = (
        f"Answer the question based on the context below.\n\n"
        f"Context:\n{formatted_context}\n\nQuestion: {question}\n\nAnswer:"
    )
    answer = llm(formatted_prompt)
    return answer

def monitor_resources():
    process = psutil.Process()
    cpu_usage = process.cpu_percent(interval=1)
    memory_usage = process.memory_info().rss / (1024 * 1024)  # in MB
    return cpu_usage, memory_usage

def get_important_facts(question):
    start_time = time.time()
    cpu_before, mem_before = monitor_resources()
    
    with mlflow.start_run():
        answer = rag_chain(question)
        
        # Log relevant metrics
        cpu_after, mem_after = monitor_resources()
        cpu_diff = cpu_after - cpu_before
        mem_diff = mem_after - mem_before
        end_time = time.time()
        response_time = end_time - start_time
        
        # Log metrics to MLflow
        mlflow.log_param("question", question)
        mlflow.log_metric("response_time", response_time)
        mlflow.log_metric("memory_usage", mem_diff)
        mlflow.log_metric("cpu_usage", cpu_diff)

        print(
            f"Response Time: {response_time:.2f} seconds, "
            f"Memory usage: {mem_diff:.2f} MB, CPU usage: {cpu_diff:.2f}%"
        )
        
    return answer

# Create a Gradio app interface
iface = gr.Interface(
    fn=get_important_facts,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs="text",
    title="Chatbot",
    description="Ask questions about the content in your PDFs",
)

# Launch the Gradio app
iface.launch(debug=True)
