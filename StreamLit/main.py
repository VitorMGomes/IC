import streamlit as st

st.title("Bot chato")

if("messages" not in st.session_state):
    st.session_state.messages = []
    

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
prompt = st.chat_input("Digite algo")

if prompt :
    st.chat_message("user").markdown(prompt)
    
    response = f"Você disse: {prompt}"
    
    with st.chat_message("assistant"):
        st.markdown(response)
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        