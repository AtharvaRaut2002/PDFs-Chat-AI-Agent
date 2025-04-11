# 📄🤖 PDFs-Chat-AI-Agent

**PDFs-Chat-AI-Agent** is an AI-powered FastAPI application that allows you to ask questions about the content of PDF documents. It reads and processes your PDFs, breaks the content into manageable chunks, creates a semantic vector store, and uses Google Gemini AI to answer questions based on the provided context.

This project was developed to learn how to build document-aware agents using LangChain and vector databases.

---

## 🔍 Features

- 🧾 Extracts text from PDFs using `PyPDF2`
- 🧠 Splits content using `LangChain`'s `RecursiveCharacterTextSplitter`
- 🧬 Generates embeddings using **Google Generative AI**
- 🗂️ Stores chunks in a local FAISS vector database
- 💬 Answers natural language questions based on PDF content
- 🌐 Built with **FastAPI** for simple API interaction

---

## 💡 Use Case

You upload a set of PDFs, and the app lets you query those documents intelligently. Useful for:

- Research paper Q&A  
- Policy document queries  
- E-book summary chats  
- and more…

---

## ⚙️ How It Works

1. Load and extract text from PDF files in the `docs/` folder
2. Split text into overlapping chunks
3. Generate embeddings using **Google Embedding API**
4. Store chunks in a FAISS vector index
5. On API call, search relevant chunks and pass them to **Gemini Pro** via LangChain to generate a contextual answer

---
