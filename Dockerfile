FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY checkpoints /app/pretrained_models

RUN pip install -e .

EXPOSE 8000

CMD ["/bin/bash"]
