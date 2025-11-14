# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for PyPDFLoader 
# libpq-dev is for PostgreSQL and may not be needed, but kept for safety.
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code and policy folder
COPY . .

# Expose the port the app will run on
EXPOSE 8000

# Command to run the application: Use start.py for reliable environment loading
CMD ["python", "start.py"]
