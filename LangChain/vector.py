from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Carrega o CSV dos holerites
df = pd.read_csv("holerites.csv", sep=";")

# Define o modelo de embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Caminho do banco vetorial
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        # Transforma a linha inteira em texto formatado
        page_content = "\n".join([
            f"{col}: {row[col]}" for col in df.columns
        ])

        document = Document(
            page_content=page_content,
            metadata={
                "nome": row["Nome Completo"],
                "email": row["Email"],
                "mes": row["Mês/Ano"]
            },
            id=str(i)
        )
        documents.append(document)
        ids.append(str(i))

# Cria (ou carrega) o vector store
vector_store = Chroma(
    collection_name="holerites_colaboradores",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Adiciona os documentos (só na primeira vez)
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# Cria o retriever para buscas semânticas
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
