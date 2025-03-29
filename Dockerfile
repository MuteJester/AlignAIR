FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY checkpoints /app/pretrained_models

RUN pip install -e .

EXPOSE 8000

# This is the key change: direct entrypoint that appends arguments to app.py
ENTRYPOINT ["python", "app.py"]

# Optional: if you want a default command, do:
# CMD ["--mode=interactive"]
