from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS  # Import CORS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os
import pickle
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')

# Define the prompt template for the QA chain
prompt_template = """
You are a car manual expert and highly knowledgeable about vehicle maintenance and operation.
Based on the following excerpt from a car manual, diagrams, and tables:
{context}
Question: {question}
If the information is not available or the question is outside the context, respond with "Sorry, I don't have much information about it."
Provide a detailed and informative answer based on the context provided.
Answer:
"""

# Initialize the QA chain with the prompt template
qa_chain = LLMChain(
    llm=ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key, max_tokens=1024),
    prompt=PromptTemplate.from_template(prompt_template)
)

# Load the FAISS index and metadata using langchain.vectorstores.FAISS
vectorstore = FAISS.load_local("faiss_index", OpenAIEmbeddings(openai_api_key=openai_api_key), allow_dangerous_deserialization=True)

def answer(question):
    print(f"Question: {question}")

    # Perform similarity search using the FAISS vectorstore
    relevant_docs = vectorstore.similarity_search(question)
    print(f"Relevant Docs: {relevant_docs}")

    context = ""
    relevant_images = []
    for d in relevant_docs:
        if d.metadata['type'] == 'text':
            context += '[text]' + d.metadata['original_content']
        elif d.metadata['type'] == 'table':
            context += '[table]' + d.metadata['original_content']
        elif d.metadata['type'] == 'image':
            context += '[image]' + d.page_content
            relevant_images.append(d.metadata['original_content'])
    
    print(f"Context: {context}")
    result = qa_chain.run({'context': context, 'question': question})
    print(f"Result: {result}")
    return result, relevant_images

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    result, relevant_images = answer(question)
    response = {'answer': result}
    
    if relevant_images:
        # Include only the first image in the response for simplicity
        response['image'] = relevant_images[0]
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
