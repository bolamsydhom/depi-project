import os
import glob
import mlflow
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from transformers import AutoTokenizer

# Specify tracking server
mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
mlflow.set_tracking_uri(mlflow_tracking_uri)

def train_model():
    # Start an MLflow run
    with mlflow.start_run():
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

        # Log parameters
        mlflow.log_param("embedding_model_name", embedding_model_name)

        # Load the LLM model
        llm_model_name = 'PubMed_Llama3.1_Based_model'
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

        # Log parameters
        mlflow.log_param("llm_model_name", llm_model_name)

        # Here you would add your model training code and log metrics
        # For example:
        # model = ...  # Train your model
        # accuracy = ...  # Calculate accuracy or other metrics
        # mlflow.log_metric("accuracy", accuracy)

        # Log the model (if applicable)
        # mlflow.sklearn.log_model(model, "model")

        # Optionally, log artifacts (e.g., plots, data files)
        # mlflow.log_artifact("path/to/artifact")

if __name__ == "__main__":
    train_model()