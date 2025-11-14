# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for PyPDFLoader (often required for PDF processing)
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    pdflib-bin \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code and the policy file
COPY . .

# Expose the port the app will run on
EXPOSE 8000

# Command to run the application using uvicorn
# The --reload flag is removed for production deployment
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]