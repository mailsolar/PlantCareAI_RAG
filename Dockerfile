# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for PyPDFLoader 
# pdflib-bin has been removed, poppler-utils is kept as essential
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
# ... (rest of the Dockerfile remains the same)
