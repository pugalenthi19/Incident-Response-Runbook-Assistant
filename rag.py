import os
import re

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from database import search_incidents

load_dotenv()

VECTOR_DB = "vectorstore"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MODEL_NAME = "llama-3.3-70b-versatile"


PROMPT = ChatPromptTemplate.from_template(
"""
You are an experienced IT Incident Response Engineer.

You have access to:

1. Official Documentation
2. Incident History

Answer ONLY using the supplied context.

If previous incidents are relevant,
mention them briefly.

If the answer is not available,
reply:

"I couldn't find this information in the knowledge base."

==========================
Documentation

{context}

==========================
Previous Incidents

{incidents}

==========================
Question

{question}

==========================
Answer
"""
)


class IncidentAssistant:

    def __init__(self):

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        self.vector_db = Chroma(
            persist_directory=VECTOR_DB,
            embedding_function=self.embeddings
        )

        self.retriever = self.vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 6,
                "fetch_k": 20,
                "lambda_mult": 0.7
            }
        )

        self.llm = ChatGroq(
            model=MODEL_NAME,
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )

        self.chain = PROMPT | self.llm

    def search_documents(self, question):

        docs = self.retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        return docs, context

    def search_database(self, question):

        keywords = [
            "docker",
            "kubernetes",
            "crashloopbackoff",
            "imagepullbackoff",
            "pod",
            "nginx",
            "linux",
            "aws",
            "dns",
            "tcp",
            "memory",
            "disk",
            "ec2",
            "iam",
            "network",
            "container"
        ]

        question = question.lower()

        keyword = None

        for word in keywords:

            if word in question:
                keyword = word
                break

        if keyword is None:
            return "No similar incidents found."

        incidents = search_incidents(keyword)

        if not incidents:
            return "No similar incidents found."

        output = []

        for row in incidents:

            output.append(
f"""
Title : {row['title']}

Category : {row['category']}

Affected Service : {row['affected_service']}

Assigned Team : {row['assigned_team']}

Severity : {row['severity']}

Symptoms : {row['symptoms']}

Root Cause : {row['root_cause']}

Resolution : {row['resolution']}

Status : {row['status']}
"""
            )

        return "\n".join(output)

    def ask(self, question):

        docs, context = self.search_documents(question)

        incidents = self.search_database(question)

        response = self.chain.invoke(
            {
                "context": context,
                "incidents": incidents,
                "question": question
            }
        )

        return {
            "answer": response.content,
            "documents": docs,
            "incidents": incidents
        }


assistant = IncidentAssistant()