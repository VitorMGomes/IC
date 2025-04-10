from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from vector import retriever
import streamlit as st

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

# Título do app
st.title("🧾 Slip Pay Agent")

# Inicializa histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada do usuário
user_input = st.chat_input("Digite sua pergunta sobre a folha de pagamento")

if user_input:
    # Mostra mensagem do usuário
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Consulta o modelo via LangChain
    with st.spinner("Consultando IA..."):
        result = qa_chain.invoke({"query": user_input})
        response = result["result"]

    # Mostra resposta do agente
    with st.chat_message("assistant"):
        st.markdown(response)

    # Salva no histórico
    st.session_state.messages.append({"role": "assistant", "content": response})
