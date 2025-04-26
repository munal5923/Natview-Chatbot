from flask import Flask, render_template, jsonify, request, Response, stream_with_context
from src.utils import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from src.prompt import *
from src.utils import simulate_streaming_response
import os
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

load_dotenv()

import os
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


embeddings = download_hugging_face_embeddings()


index_name = "niftybot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


llm=ChatGroq(groq_api_key=GROQ_API_KEY,model_name="deepseek-r1-distill-llama-70b",temperature=0.3)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    clean_response = re.sub(r"<think>.*?</think>", "", response['answer'], flags=re.DOTALL).strip()
    print("Response : ", response["answer"])
    return str(clean_response)



# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port= 5000, debug= True)
