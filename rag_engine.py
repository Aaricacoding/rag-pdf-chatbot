# rag_engine.py — Core RAG pipeline using LangChain + FAISS + HuggingFace
# This file handles: PDF loading → text splitting → embedding → vector indexing → QA chain

# LangChain's PDF loader — reads PDF pages and converts them into Document objects
from langchain_community.document_loaders import PyPDFLoader

# RecursiveCharacterTextSplitter — splits long text into overlapping chunks
# so context isn't lost at chunk boundaries when we search later
from langchain.text_splitter import RecursiveCharacterTextSplitter

# HuggingFaceEmbeddings — converts text chunks into semantic float vectors
# Uses sentence-transformers/all-MiniLM-L6-v2 — free, fast, no API key needed
from langchain_community.embeddings import HuggingFaceEmbeddings

# FAISS — Facebook AI Similarity Search, a local vector database
# Stores embeddings and lets us find the most relevant chunks by cosine similarity
from langchain_community.vectorstores import FAISS

# HuggingFacePipeline — wraps a HuggingFace transformers pipeline
# so LangChain can treat it as a standard LLM object
from langchain_community.llms import HuggingFacePipeline

# RetrievalQA — LangChain chain that connects: retriever → prompt → LLM → answer
from langchain.chains import RetrievalQA

# transformers pipeline — loads a local open-source LLM (no OpenAI key needed)
from transformers import pipeline

import os  # standard os module for file path utilities


def load_and_index_pdf(pdf_path: str) -> FAISS:
    """
    Loads a PDF from disk, splits it into chunks, embeds each chunk,
    and stores everything in a FAISS vector index for fast similarity search.

    Args:
        pdf_path (str): Absolute or relative path to the PDF file

    Returns:
        FAISS: An indexed vector store ready for .similarity_search() or .as_retriever()
    """

    # Step 1: Load PDF — each page becomes a Document(page_content=str, metadata={page: N})
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()  # list of Document objects, one per PDF page

    # Step 2: Split documents into smaller chunks
    # chunk_size=500 chars keeps each chunk focused on one topic
    # chunk_overlap=50 chars so sentences that span chunk boundaries aren't lost
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,    # max characters per chunk — tune lower for precision, higher for context
        chunk_overlap=50   # overlap between consecutive chunks to preserve context
    )
    chunks = splitter.split_documents(documents)  # returns a larger list of smaller Documents

    # Step 3: Load the embedding model
    # all-MiniLM-L6-v2 is ~80MB, runs on CPU, produces 384-dimensional vectors
    # model_kwargs device='cpu' — ensures it works on any machine without GPU
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # change to "cuda" if you have an NVIDIA GPU
    )

    # Step 4: Build FAISS index
    # This embeds every chunk and stores (vector, chunk_text) pairs
    # from_documents handles the embed + index in one call
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store  # ready for querying


def build_qa_chain(vector_store: FAISS) -> RetrievalQA:
    """
    Builds a full RetrievalQA chain by combining:
    - A FAISS retriever (finds relevant chunks)
    - A local HuggingFace LLM (generates the answer)

    Args:
        vector_store (FAISS): The indexed vector store from load_and_index_pdf()

    Returns:
        RetrievalQA: A callable chain — use as chain({"query": "your question"})
    """

    # Step 1: Load a lightweight open-source LLM locally
    # google/flan-t5-base is ~250MB, good at Q&A, works without any API key
    # task="text2text-generation" is suited for question → answer style tasks
    # max_new_tokens=256 caps response length so it runs fast on CPU
    hf_pipeline = pipeline(
        "text2text-generation",       # generation task type
        model="google/flan-t5-base",  # small but effective instruction-following model
        max_new_tokens=256,           # maximum tokens in the generated answer
        do_sample=False               # deterministic output — same question → same answer
    )

    # Step 2: Wrap the HF pipeline so LangChain recognizes it as an LLM
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # Step 3: Convert the vector store into a retriever
    # k=3 means: for each question, fetch the 3 most relevant text chunks
    # increasing k gives more context but also more noise
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}  # number of top chunks to retrieve
    )

    # Step 4: Create the RetrievalQA chain
    # chain_type="stuff" = concatenate all retrieved chunks into one prompt
    # Works well when chunks are small (<=500 chars each)
    # return_source_documents=True so we can show which pages the answer came from
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,                          # the language model for answer generation
        chain_type="stuff",               # prompt strategy: stuff all chunks together
        retriever=retriever,              # the FAISS-based retriever
        return_source_documents=True      # include source chunks in the output dict
    )

    return qa_chain  # caller can now do: result = qa_chain({"query": "What is X?"})
