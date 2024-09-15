from flask import Flask, request, jsonify
import subprocess
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama


app = Flask(__name__)

# Step 1: Initialize Knowledge Base and FAISS Index
knowledge_base_file = 'knowledge_base.txt'

# Load the knowledge base
with open(knowledge_base_file, 'r') as f:
    knowledge_base = f.readlines()

# Encode the knowledge base using Sentence Transformers
model = SentenceTransformer('all-MiniLM-L6-v2')
knowledge_embeddings = model.encode(knowledge_base, convert_to_tensor=True)

# Create a FAISS index for efficient retrieval
index = faiss.IndexFlatL2(knowledge_embeddings.shape[1])
index.add(np.array(knowledge_embeddings))

# Function to retrieve relevant knowledge
def retrieve_knowledge(query, top_k=2):
    query_embedding = model.encode([query])
    _, top_indices = index.search(query_embedding, top_k)
    return [knowledge_base[idx] for idx in top_indices[0]]

# Step 2: Integrate Ollama's LLAMA3 with LangChain
# Initialize Ollama with LangChain
llama3_llm = Ollama(model="llama3")

# Define a prompt template for LangChain
prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template="Given the context: {context}\nAnswer the query: {query}"
)

# Create an LLM Chain using LangChain
llama_chain = LLMChain(llm=llama3_llm, prompt=prompt_template)


# Assuming ollama is imported and instantiated correctly
def ungrounded_response_with_ollama(query):
    try:

        # Use the ollama instance to generate a response
        result = llama3_llm.invoke(model='llama3', input=query)
        
        # Assume result is a simple text response
        return result
        
    except Exception as e:
        return f"Error in processing the response from Ollama: {str(e)}"


from flask import Flask, request, jsonify
import asyncio

@app.route('/generate', methods=['POST'])
async def generate_response():
    data = request.json
    query = data.get('query', '')

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Run the responses asynchronously
    grounded_resp = await asyncio.to_thread(grounded_response_with_langchain, query)
    ungrounded_resp = await asyncio.to_thread(ungrounded_response_with_ollama, query)

    return jsonify({
        "query": query,
        "grounded_response": grounded_resp,
        "ungrounded_response": ungrounded_resp
    })


# Function to generate a grounded response using LangChain with Ollama
def grounded_response_with_langchain(query):
    # Retrieve relevant knowledge
    knowledge = retrieve_knowledge(query)
    context = " ".join(knowledge)
    
    # Use LangChain to generate a grounded response with the context
    response = llama_chain.invoke({"context": context, "query": query})
    return response



# Step 4: Run the Flask App
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
