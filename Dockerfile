# Start from a lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire repository into the container
COPY . .

# Install the current repository in developer mode
RUN pip install -e .

# Set the entrypoint to run the interactive CLI tool
ENTRYPOINT ["python", "app.py"]
