# Start from a lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire repository into the container
COPY . .

COPY checkpoints /app/pretrained_models


# Install the current repository in developer mode
RUN pip install -e .

EXPOSE 8000

ENV APP_FILE="app.py"

# Entrypoint that runs based on the selected app file
ENTRYPOINT ["sh", "-c", "python ${APP_FILE}"]