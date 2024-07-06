import streamlit as st
import PyPDF2
from langchain_community.embeddings import JinaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
load_dotenv()
st.title("📄 Chat with your PDF Document using Groq")

# Initialize session state variables
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = []


st.write("API keys are set. You can now upload a PDF file.")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:

    pdf = PyPDF2.PdfReader(uploaded_file)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(pdf_text)

        # Create metadata for each chunk
        metadatas = [{"source": f"{i}-{uploaded_file.name}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    embeddings = JinaEmbeddings()
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas)

    # Initialize message history for conversation
    if "message_history" not in st.session_state:
        st.session_state.message_history = ChatMessageHistory()

    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=st.session_state.message_history,
        return_messages=True,
    )

    # Initialize Groq LLM
    llm_groq = ChatGroq(model_name='mixtral-8x7b-32768')

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_groq,
        chain_type="stuff",
        retriever=docsearch.as_retriever(k=7),
        memory=memory,
        return_source_documents=True,
    )

    # Store the chain in the session state
    st.session_state.chain = chain

    st.write(f"Processing `{uploaded_file.name}` done. You can now ask questions!")

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Display sources in the sidebar
with st.sidebar:
    st.header("Sources")
    for source in st.session_state.sources:
        st.write(source)

# Text input for user queries
if prompt := st.chat_input(placeholder="Ask a question about the PDF"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    chain = st.session_state.chain
    if chain is not None:
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            res = chain.invoke(prompt)
            answer = res["answer"]
            source_documents = res["source_documents"]

            response_text = answer
            st.session_state.sources = []

            if source_documents:
                response_text += "\n\nSources:\n"
                for idx, doc in enumerate(source_documents):
                    short_version = doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
                    st.session_state.sources.append(f"Source {idx + 1}: {short_version}")
                    response_text += f"{idx + 1}. {short_version}\n"

            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.write(response_text)