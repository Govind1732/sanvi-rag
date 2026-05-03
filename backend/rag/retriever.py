from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_retriever():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    db = FAISS.load_local("db", embeddings)
    return db.as_retriever(search_kwargs={"k": 3})