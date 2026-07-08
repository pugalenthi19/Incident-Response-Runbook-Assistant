import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


DOCS_PATH = "data/docs"
RUNBOOK_PATH = "data/runbook"
VECTOR_DB = "vectorstore"


def load_pdf_folder(folder_path):

    documents = []

    skipped_files = []

    pdf_files = []

    for root, _, files in os.walk(folder_path):

        for file in files:

            if file.lower().endswith(".pdf"):

                pdf_files.append(os.path.join(root, file))

    print(f"\nFound {len(pdf_files)} PDFs in {folder_path}")

    for pdf in pdf_files:

        try:

            loader = PyPDFLoader(pdf)

            docs = loader.load()

            documents.extend(docs)

            print(f"Loaded : {os.path.basename(pdf)}")

        except Exception as e:

            skipped_files.append(pdf)

            print(f"Skipped : {os.path.basename(pdf)}")

            print(f"Reason : {e}")

    return documents, skipped_files


def load_documents():

    print("\nLoading Documentation...\n")

    docs, skipped_docs = load_pdf_folder(DOCS_PATH)

    print("\nLoading Runbooks...\n")

    runbooks, skipped_runbooks = load_pdf_folder(RUNBOOK_PATH)

    documents = docs + runbooks

    skipped = skipped_docs + skipped_runbooks

    print("\n--------------------------------")

    print(f"Total Pages Loaded : {len(documents)}")

    print(f"Skipped PDFs : {len(skipped)}")

    print("--------------------------------\n")

    return documents


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=200

    )

    chunks = splitter.split_documents(documents)

    print(f"Total Chunks : {len(chunks)}")

    return chunks


def load_embeddings():

    return HuggingFaceEmbeddings(

        model_name="sentence-transformers/all-MiniLM-L6-v2"

    )


def create_vector_database():

    if os.path.exists(VECTOR_DB):

        print("Vector Database Already Exists.")

        return

    documents = load_documents()

    chunks = split_documents(documents)

    embeddings = load_embeddings()

    print("\nCreating Chroma Database...\n")

    Chroma.from_documents(

        documents=chunks,

        embedding=embeddings,

        persist_directory=VECTOR_DB

    )

    print("\nVector Database Created Successfully.\n")


if __name__ == "__main__":

    create_vector_database()