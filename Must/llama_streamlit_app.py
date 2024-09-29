import streamlit as st
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

# Step 1: Initialize Knowledge Base and FAISS Index
knowledge_base_file = 'knowledge_base.txt'

# Load the knowledge base
with open(knowledge_base_file, 'r', encoding='utf-8') as f:
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

# Function to generate an ungrounded response using Ollama's Python interface
def ungrounded_response_with_ollama(query):
    try:
        result = llama3_llm.invoke(query)
        return result
    except Exception as e:
        return f"Error in processing the response from Ollama: {str(e)}"

# Function to generate a grounded response using LangChain with Ollama
def grounded_response_with_langchain(query):
    knowledge = retrieve_knowledge(query)
    context = " ".join(knowledge)
    
    # Use LangChain to generate a grounded response with the context
    response = llama_chain.invoke({"context": context, "query": query})
    return {
        "context": context,
        "query": query,
        "text": response
    }

# Step 3: Streamlit Interface
st.title("LLM Query Interface")
st.write("Generate responses using LLAMA3 with and without context.")

query = st.text_input("Enter your query:")

if st.button("Generate Response"):
    if query:
        # Generate the responses
        with st.spinner('Generating responses...'):
            grounded_resp = grounded_response_with_langchain(query)
            ungrounded_resp = ungrounded_response_with_ollama(query)
        
        # Display the Grounded Response with visual improvements
        st.subheader("Grounded Response")
        
        # Display context, query, and response text separately
        st.markdown("**Context Retrieved from Knowledge Base:**")
        st.code(grounded_resp["context"])

        st.markdown("**Query:**")
        st.code(grounded_resp["query"])

        st.markdown("**Response:**")
        st.markdown(grounded_resp["text"])

        # Display the Ungrounded Response
        st.subheader("Ungrounded Response")
        st.markdown(f"**Response**: {ungrounded_resp}")
    else:
        st.error("Please enter a query.")
