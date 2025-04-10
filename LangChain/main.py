from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from vector import retriever  # Importa o retriever criado no vector.py

# Modelo de linguagem
model = OllamaLLM(model="llama3.2", streaming=False)

# Prompt base
template = """
Você é responsável pela folha de pagamento de uma empresa.

Aqui estão os dados relevantes:
{context}

Baseando-se nesses dados, responda à seguinte pergunta: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Cria o chain com recuperação vetorial
qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# Loop de perguntas
while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (Press Q to quit): ")
    print("\n\n")

    if question.lower() == "q":
        break

    result = qa_chain.invoke({"query": question})
    print(result)
