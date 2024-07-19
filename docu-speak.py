import streamlit as st
import PyPDF2
from langchain_community.embeddings import JinaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import logging
import sys
import uuid

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Streamlit setup
st.title("📄 Chat with your PDF Document using Groq")

# Fetch API keys
JINA_API_KEY = st.secrets["JINA_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Generate a unique session ID for each user
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Unique session keys
session_id = st.session_state.session_id
#print(f"Session ID: {session_id}")
chain_key = f"chain_{session_id}"
messages_key = f"messages_{session_id}"
sources_key = f"sources_{session_id}"
history_key = f"history_{session_id}"

# Initialize session state variables
if chain_key not in st.session_state:
    st.session_state[chain_key] = None
if messages_key not in st.session_state:
    st.session_state[messages_key] = []
if sources_key not in st.session_state:
    st.session_state[sources_key] = []

# Function to process the uploaded PDF file
def process_pdf(uploaded_file):
    pdf = PyPDF2.PdfReader(uploaded_file)
    pdf_text = "".join([page.extract_text() for page in pdf.pages])

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    texts = text_splitter.split_text(pdf_text)

    # Create metadata for each chunk
    metadatas = [{"source": f"{i}-{uploaded_file.name}-pl"} for i in range(len(texts))]
    
    # Create a Chroma vector store
    embeddings = JinaEmbeddings(jina_api_key=JINA_API_KEY)
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas)

    return docsearch

# Function to initialize the chain
def initialize_chain(docsearch):
    llm_groq = ChatGroq(model_name='llama3-70b-8192', api_key=GROQ_API_KEY, temperature=0.3)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=st.session_state[history_key],
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_groq,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    return chain

# UI for file upload
st.write("API keys are set. You can now upload a PDF file.")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    try:
        docsearch = process_pdf(uploaded_file)
        if history_key not in st.session_state:
            st.session_state[history_key] = ChatMessageHistory()
        st.session_state[chain_key] = initialize_chain(docsearch)
        st.write(f"Processing `{uploaded_file.name}` done. You can now ask questions!")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        logging.error(f"Error processing PDF: {e}")

# Display chat history
for msg in st.session_state[messages_key]:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user queries
if prompt := st.chat_input(placeholder="Ask a question about the PDF"):
    st.session_state[messages_key].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    chain = st.session_state[chain_key]
    if chain is not None:
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            res = chain.invoke(prompt)
            answer = res["answer"]
            source_documents = res["source_documents"]

            response_text = answer
            st.session_state[sources_key] = []

            if source_documents:
                for idx, doc in enumerate(source_documents):
                    short_version = doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
                    st.session_state[sources_key].append(f"Source {idx + 1}: {short_version}")
                
            with st.sidebar:
                st.header("Sources")
                for source in st.session_state[sources_key]:
                    st.markdown(source)
                st.session_state[sources_key] = []

            st.session_state[messages_key].append({"role": "assistant", "content": response_text})
            st.write(response_text)

# Clear chat history
if st.button("Clear chat history"):
    st.session_state[messages_key] = []
    st.session_state[sources_key] = []
    st.session_state[chain_key] = None
    st.session_state[history_key] = ChatMessageHistory()
    del uploaded_file
    del st.session_state['session_id'] 
    docsearch = None
    st.write("Chat history cleared.")
    with st.sidebar:
        st.empty()
    st.empty()
