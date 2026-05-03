from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from .retriever import get_retriever

def get_rag_chain():
    retriever = get_retriever()

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",   # fast + cheap
        temperature=0
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    return chain