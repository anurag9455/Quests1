# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies for FAISS and other requirements
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libomp-dev

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit will run on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "llama_streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
