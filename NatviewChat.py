import streamlit as st
import os
from langchain_groq import ChatGroq
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader,PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
# import openai

from dotenv import load_dotenv
load_dotenv()
## load the GROQ API Key
# os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
# os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
# os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")

groq_api_key = st.secrets["GROQ_API_KEY"]
hf_token = st.secrets["HF_TOKEN"]

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


groq_api_key=os.getenv("GROQ_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="deepseek-r1-distill-llama-70b")

session_id="default_session"
## statefully manage chat history

if 'store' not in st.session_state:
    st.session_state.store={}


embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
loader=PyPDFLoader("NatviewWebsite.pdf") ## Data Ingestion step
docs=loader.load() ## Document Loading

print(f"Total documents loaded: {len(docs)}")
text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=200)
final_documents=text_splitter.split_documents(docs[:50])
vectors=FAISS.from_documents(final_documents,embeddings)
retriever = vectors.as_retriever()


system_prompt=(
    "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to"
    "answer the question. If you don't know the answer, say that you don't know. keep the answer concise."
    "Answer the questions in a polite manner and based on the provided context only."
    "Please provide the most accurate respone based on the question"
    "\n\n"
    "{context}"
)



contextualize_q_system_prompt=(
    "Given a chat history and the latest user question"
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as it is."
    "When the user indicates the dont want you to combine their question with history"
    "do not use it"
    "While using the chat history only make sure the question asked has a very strong relation with"
    "the history is not go straight to the answer"
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

## Answer question

qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

def get_session_history(session:str)->BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id]=ChatMessageHistory()
    return st.session_state.store[session_id]

conversational_rag_chain=RunnableWithMessageHistory(
    rag_chain,get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

st.title("Natview Chatbot")

user_input = st.text_input("Your question:")
if user_input:
    session_history=get_session_history(session_id)
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={
            "configurable": {"session_id":session_id}
        },  # constructs a key "abc123" in `store`.
    )

    import re
    # Extract the answer and remove any "<think>...</think>" parts
    clean_response = re.sub(r"<think>.*?</think>", "", response['answer'], flags=re.DOTALL).strip()
    
    st.write(clean_response)



# st.title("RAG Document Q&A With Groq And Lama3")

# user_prompt=st.text_input("Enter your query from the research paper")

# # if st.button("Document Embedding"):
# #     create_vector_embedding()
# #     st.write("Vector Database is ready")

# import time

# if user_prompt:
#     document_chain=create_stuff_documents_chain(llm,prompt)
#     retriever=vectors.as_retriever()
#     retrieval_chain=create_retrieval_chain(retriever,document_chain)

#     start=time.process_time()
#     response=retrieval_chain.invoke({'input':user_prompt})
#     print(f"Response time :{time.process_time()-start}")

#     st.write(response['answer'])

#     ## With a streamlit expander
#     with st.expander("Document similarity Search"):
#         for i,doc in enumerate(response['context']):
#             st.write(doc.page_content)
#             st.write('------------------------')





