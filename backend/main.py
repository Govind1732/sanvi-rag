from fastapi import FastAPI, UploadFile, File
from rag.ingest import ingest_pdf
from rag.chain import get_rag_chain
import shutil

app = FastAPI()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    path = f"data/{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"message": ingest_pdf(path)}

@app.get("/ask")
def ask(query: str):
    chain = get_rag_chain()
    response = chain.run(query)
    return {"answer": response}