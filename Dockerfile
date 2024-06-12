# Use the nvidia/cuda base image with CUDA 11 and libcudnn8
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Install Python 3.9 and pip
RUN apt-get update && \
    apt-get install -y python3.9 python3.9-dev python3.9-distutils && \
    apt-get install -y wget curl git && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.9 get-pip.py && \
    rm get-pip.py

# Create a symbolic link to use 'python' command for Python 3.9
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Install other necessary Python packages
# RUN pip install --no-cache-dir numpy scipy pandas

# Setup work directory
RUN mkdir -p /work
WORKDIR /work
COPY . .


RUN pip install -e . 
RUN pip install omegaconf


# Clean up unnecessary packages to reduce image size
RUN apt-get purge -y --auto-remove wget curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*




# Set the default command to bash
CMD ["/bin/bash"]

# docker build -t alignair:latest .
# docker run --rm -it --gpus all alignair:latest 
# alignair_predict