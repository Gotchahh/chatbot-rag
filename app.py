import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

st.title("Chatbot RAG su PDF")
st.write("Carica un PDF e fai domande sul suo contenuto")

uploaded_file = st.file_uploader("Carica il tuo PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Elaboro il documento..."):
        loader = PyMuPDFLoader("temp.pdf")
        documents = loader.load()
        seen = set()
        unique_documents = []
        for doc in documents:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_documents.append(doc)
        documents = unique_documents

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=150
        )
        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        vectorstore = Chroma.from_documents(chunks, embeddings)

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=st.secrets["GROQ_API_KEY"]
        )

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 10}
            )
        )

    st.success(f"PDF caricato! {len(chunks)} chunks creati.")
    st.divider()
    domanda = st.text_input("Fai una domanda sul documento:")

    if domanda:
        with st.spinner("Sto cercando la risposta..."):
            risposta = chain.run(domanda)
            st.write("**Risposta:**", risposta)