import os
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain.chains import ConversationalRetrievalChain
import nltk 
import time 
nltk.download('punkt')
import streamlit as st

# pip install streamlit-chat  
from streamlit_chat import message
import os

# openai_api_key = st.secrets["key_openai"]
print("hellooo:   ")
print(st.secrets["key_openai"])

key_2=st.secrets["key_openai"]
os.environ['OPENAI_API_KEY'] = key_2
openai.api_key = os.getenv("OPENAI_API_KEY")

persist_directory = 'persist_directory'
embeddings=OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
doc_search = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
chain = ConversationalRetrievalChain.from_llm(llm=OpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff", retriever=doc_search.as_retriever(), return_source_documents=True)

# from dotenv import load_dotenv
# load_dotenv() 

# key_1=os.getenv("key_openai")
# os.environ['OPENAI_API_KEY'] = key_1
# os.environ["OPENAI_API_KEY"] == st.secrets["key_openai"],

# persist_directory = 'persist_directory'
# embeddings=OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
# doc_search= Chroma(persist_directory=persist_directory, embedding_function=embeddings)
# chain=ConversationalRetrievalChain.from_llm(llm=OpenAI(model_name="gpt-3.5-turbo"),chain_type="stuff",retriever=doc_search.as_retriever(), return_source_documents=True) 

def generate_response(prompt):
    chatHistory=[]
    for i in range(len(st.session_state.generated)):
        chatHistory.append((st.session_state.past[i],st.session_state.generated[i]))
    beg= time.time()
    response=chain({"question": prompt, "chat_history": chatHistory})
    result=response["answer"]
    try :
        path=dict(response["source_documents"][0])["metadata"]["source"]
        file_name=path.split("/")[-1]
    except : 
        file_name=" "
    end= time.time()
    temps=round(end-beg,2)
    return result + "\n" + 'source: '+ file_name.replace("txt","pdf") +"\n" +f" temps de traitement :{temps}"

from PIL import Image

#Creating the chatbot interface
# Afficher le titre avec une police et un alignement personnalisés
st.sidebar.markdown("""
    <h1 style='font-family: Cascadia Mono; text-align: left; margin-top: -70px;margin-bottom: 25px;'>AmpliBot</h1>
    """, unsafe_allow_html=True)

image = Image.open('logo_banque.png')
width, height = image.size
new_width = 240
new_height = int(height * new_width / width)
resized_image = image.resize((new_width, new_height))
st.sidebar.image(resized_image)

# Afficher le texte sur plusieurs lignes
st.sidebar.markdown("""<p style='font-family: Cascadia Mono; text-align: center;margin-top: 10px;line-height: 200%'>
    Découvrez comment un chatbot innovant pourrait révolutionner l'usage du système d'information bancaire Amplitude, 
    optimisant les réponses et améliorant l'efficacité. 
    Un gain de temps précieux vous attend, 
    plongez-vous dans ce projet captivant!
    </p>""",unsafe_allow_html=True)


st.markdown("""
    <h3 style='font-family: Cascadia Mono; text-align: center; margin-top: -60px;margin-bottom: 25px;'>AmpliBot : L'Assistant Intelligent d'Amplitude</h3>
    """, unsafe_allow_html=True)

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

# We will get the user's input by calling the get_text function
def get_text():
    input_text = st.text_input(
        "",
        value="Module cheque",
        key="input"
    )
    return input_text

# Appliquer le style personnalisé
st.markdown("""
    <p style='font-family: Cascadia Mono; text-align: left;margin-top: 50px;margin-bottom: -45px;'>Saisir votre question</p>
    """, unsafe_allow_html=True)

user_input = get_text()
if user_input:
    # Créer une mise en page en colonne
    col1, col2 = st.columns([6, 1])
    
    # Ajouter une marge à droite pour déplacer le bouton à droite
    col2.empty()
    col2.markdown('<style>div.stButton > button:first-child {margin-left: auto;margin-right: 0;margin-top: -20px;background-color: #ef6110;color: #ffffff}</style>', unsafe_allow_html=True)
    
    with col2:
        if st.button('Envoyer'):
            output = generate_response(user_input)

            # Stocker la sortie
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)
if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
