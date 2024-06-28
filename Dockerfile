# Use an official Python runtime as a parent image
FROM python:3.10.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements.txt and example.env initially to install dependencies
COPY requirements.txt example.env /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app

# Expose the port number that Streamlit listens on
EXPOSE 8501

# Set environment variables
# Adjust as needed based on your environment variable name
ENV GOOGLE_API_KEY=""

# Load environment variables from example.env file
# Adjust this line according to your actual environment variable setup
RUN dos2unix /app/example.env

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py"]
