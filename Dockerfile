# Use an official TensorFlow runtime as a parent image
FROM python:3.9-alpine3.20

# Set the working directory in the container
WORKDIR /usr/AlignAIRR

# Copy the current directory contents into the container at /usr/AlignAIRR
COPY . .

# Set the virtual environment path
ENV VIRTUAL_ENV=/usr/AlignAIRR/AlignAIR_ENV
ENV PATH="$VIRTUAL_ENV/Scripts:$PATH"


# Install any needed packages specified in requirements.txt
#RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run main.py when the container launches
CMD ["python", "main.py"]
