# %%
!pip install langchain_groq langchain_core langchain_community


# %%
from langchain_groq import ChatGroq
llm = ChatGroq(
    temperature = 0,
    groq_api_key = "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",
    model_name = "llama-3.3-70b-versatile"
)
result = llm.invoke("Who is lord Ram?")
print(result.content)

# %%
!pip install pypdf

# %%
!pip install chromadb

# %%
!pip install sentence_transformers


# %%
pip install langchain


# %%
exit()


# %%
pip install langchain-community


# %%
exit()

# %%
pip install --upgrade langchain


# %%
pip install langchain-groq


# %%
exit()


# %%
import os
data_path = "./data"

if not os.path.exists(data_path):
    os.makedirs(data_path)
    print(f"Directory '{data_path}' created!")
else:
    print(f"Directory '{data_path}' already exists.")


# %%
exit()


# %%
pip install -U langchain-huggingface


# %%
pip install ipywidgets


# %%
exit()


# %%
pip install chromadb


# %%
exit()


# %%
exit()

# %%
pip install -U langchain-huggingface


# %%
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def initialize_llm():
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        temperature=0,
        groq_api_key="enter_your_api_key_here",
        model_name="llama-3.3-70b-versatile"
    )
    return llm

def create_vector_db(data_path, persist_path):
    # Check if the directory exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Directory not found: '{data_path}'")

    # Load PDF documents
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Initialize embeddings and create vector DB
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory=persist_path)
    vector_db.persist()

    print("ChromaDB created and data saved")

    return vector_db

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_templates = """You are a compassionate mental health chatbot. Respond thoughtfully to the following question:
    {context}
    User: {question}
    Chatbot: """
    PROMPT = PromptTemplate(template=prompt_templates, input_variables=['context', 'question'])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

def main():
    print("Initializing Chatbot.........")
    llm = initialize_llm()

    data_path = "./data"  # Adjust this path to point to your .pdf files directory
    persist_path = "./chroma_db"

    # Ensure the vector database is created or loaded
    if not os.path.exists(persist_path):
        print("Creating vector database...")
        vector_db = create_vector_db(data_path, persist_path)
    else:
        print("Loading existing vector database...")
        embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vector_db = Chroma(persist_directory=persist_path, embedding_function=embeddings)

    # Set up QA chain
    qa_chain = setup_qa_chain(vector_db, llm)

    # Chat loop
    while True:
        query = input("\nHuman: ")
        if query.lower() == "exit":
            print("Chatbot: Take care of yourself, Goodbye!")
            break
        response = qa_chain.run(query)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()


# %%
!pip install gradio

# %%
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import gradio as gr
from langchain_groq import ChatGroq


# Initialize the Language Model
def initialize_llm():
    llm = ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama-3.3-70b-versatile"
    )
    return llm


# Create the Vector Database
def create_vector_db(data_path, persist_path):
    if not os.path.exists(data_path):
        print(f"Directory not found: '{data_path}', creating it now...")
        os.makedirs(data_path)
        print(f"Please add PDF files to the directory: {data_path} and re-run the script.")
        return None

    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        print(f"No PDF files found in directory: {data_path}")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory=persist_path)
    vector_db.persist()

    print("ChromaDB created and data saved.")
    return vector_db


# Setup the RetrievalQA Chain
def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_templates = """You are a compassionate mental health chatbot. Respond thoughtfully to the following question:
    {context}
    User: {question}
    Chatbot: """
    PROMPT = PromptTemplate(template=prompt_templates, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain


# Chatbot Response Handler
def chatbot_response(user_input, history=[]):
    if not user_input.strip():
        history.append({"role": "assistant", "content": "Please provide a valid input."})
        return history
    response = qa_chain.run(user_input)
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})
    return history


# Main Code
print("Initializing Chatbot...")
llm = initialize_llm()

db_path = "./chroma_db"
data_path = "./data"

if not os.path.exists(db_path):
    print("Creating vector database...")
    vector_db = create_vector_db(data_path, db_path)
    if vector_db is None:
        print(f"Please add PDF files to '{data_path}' and re-run the script.")
        exit(1)
else:
    print("Loading existing vector database...")
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

qa_chain = setup_qa_chain(vector_db, llm)

# Gradio UI
with gr.Blocks(theme="Respair/Shiki@1.2.1") as app:
    gr.Markdown("# 🧠 Mental Health Chatbot 🤖")
    gr.Markdown("A compassionate chatbot designed to assist with mental well-being. Please note: For serious concerns, contact a professional.")

    # Create the Chatbot component
    chatbot = gr.Chatbot(type="messages")
    
    # Add Textbox for user input and Button for sending
    with gr.Row():
        user_input = gr.Textbox(label="Your Message", placeholder="Type your message here...")
        send_button = gr.Button("Send")

    # Connect components to the chatbot response function
    send_button.click(chatbot_response, [user_input, chatbot], chatbot)

    gr.Markdown("This chatbot provides general support. For urgent issues, seek help from licensed professionals.")

# Launch the Gradio app
app.launch()

# %%



