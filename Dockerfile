FROM centos:7

# Install development tools and dependencies, including GCC
RUN yum -y groupinstall "Development tools" \
    && yum -y install wget zlib-devel bzip2-devel openssl-devel ncurses-devel libxml2-devel libxslt-devel readline-devel tk-devel gdbm-devel db4-devel libpcap-devel xz-devel libjpeg-devel \
    && yum -y install gcc gcc-c++

# Download and make the CUDA installer executable
# RUN wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run -O /tmp/cuda_installer.run \
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo -O /tmp/cuda_installer.run \
    && chmod +x /tmp/cuda_installer.run
# Install CUDA toolkit silently
RUN /tmp/cuda_installer.run --toolkit --silent

RUN wget https://developer.download.nvidia.com/compute/machine-learning/repos/rhel7/x86_64/libcudnn8-8.0.4.30-1.cuda11.1.x86_64.rpm
RUN rpm -i libcudnn8-8.0.4.30-1.cuda11.1.x86_64.rpm
# RUN yum install libcudnn8-8.0.4.30-1.cuda11.1.x86_64.rpm
RUN yum -y install libcudnn8

# Clean up the installer
RUN rm /tmp/cuda_installer.run

# Remove development tools and clean up packages
RUN yum -y groupremove "Development tools" \
    && yum -y remove wget zlib-devel bzip2-devel openssl-devel ncurses-devel libxml2-devel libxslt-devel readline-devel tk-devel gdbm-devel db4-devel libpcap-devel xz-devel libjpeg-devel \
    && yum clean all

# Set environment variables for Anaconda
ENV PATH="/opt/conda/bin:$PATH"
ARG PYTHON_VERSION=3.8

# Install Anaconda
RUN yum check-update || true \
    && yum -y groupinstall "Development tools" \
    && yum -y install wget bzip2 \
    # && apt-get install -y wget bzip2 \
    && wget --quiet https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh -O /tmp/anaconda.sh \
    && /bin/bash /tmp/anaconda.sh -b -p /opt/conda \
    && rm /tmp/anaconda.sh \
    && conda init \
    && yum -y groupremove "Development tools" \
    && yum -y remove wget bzip2 \
    && yum clean all

# Create a new Anaconda environment with Python 3.8
RUN conda create -n python38 python=$PYTHON_VERSION

# Activate the environment
SHELL ["conda", "run", "-n", "python38", "/bin/bash", "-c"]


RUN mkdir -p /work
WORKDIR /work
COPY . .

# RUN conda activate python38
RUN echo "conda activate python38" >> ~/.bashrc
ENV PATH /opt/conda/envs/myenv/bin:$PATH
RUN pip install -r requirements.txt
RUN conda install cudatoolkit
RUN conda install cudnn
# RUN pip install -r requirements.txt
RUN rm -rf /opt/conda/envs/python38/lib/python3.8/site-packages/airrship/data
RUN cp -r data /opt/conda/envs/python38/lib/python3.8/site-packages/airrship/


# Start a Bash shell by default
CMD ["/bin/bash"]
