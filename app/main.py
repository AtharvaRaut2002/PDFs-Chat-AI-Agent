from fastapi import FastAPI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

@app.get("/")
def read_root():
    ans = user_input("What is CALM? Answer in 10 words?")
    return {"answer": ans}


def get_pdf_text(pdf_docs, base_path="docs"):
    text_parts = []
    for pdf in pdf_docs:
        try:
            pdf_path = os.path.join(base_path, pdf)
            print(pdf_path)
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        except Exception as e:
            print(f"Error processing {pdf}: {e}")
    
    return ''.join(text_parts)

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.3)
    print ("model ", model)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    print ("prompt ", prompt)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    print ("chain ", chain)

    return chain



def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Load the FAISS index
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        # Perform similarity search
        docs = new_db.similarity_search(user_question)
        # Get the conversational chain
        chain = get_conversational_chain()

        # Generate the response
        response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        print("Response ", response)
        return response  # Return the result

    except Exception as e:
        # Catch any exception and return a custom message
        return {"error": f"An error occurred: {str(e)}"}


def main():
     # Define the folder path
    folder_path = "docs"
    # Get all PDF files in the folder
    pdf_files = [file for file in os.listdir(folder_path) if file.endswith(".pdf")]
    
    text = get_pdf_text(pdf_files)
    chunck = get_text_chunks(text)
    
    get_vector_store(chunck)


# Run the main function by default
if __name__ == "__main__":
    main()
