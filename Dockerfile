# Use an official Python runtime as a parent image
FROM python:3.10.12

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements.txt and example.env initially to install dependencies
COPY requirements.txt example.env google_api_key.env /app/ 

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8501


ENV NAME RAG-demo

CMD ["streamlit", "run", "app.py"]
