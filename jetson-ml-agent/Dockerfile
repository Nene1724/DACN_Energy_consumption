FROM balenalib/jetson-nano-ubuntu:focal

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    python3-pip \
    python3-numpy \
    pkg-config \
    libhdf5-serial-dev \
    hdf5-tools \
    libhdf5-dev \
    zlib1g-dev \
    zip \
    libjpeg8-dev \
    liblapack-dev \
    libblas-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

COPY requirements.txt ./

# Install Python packages
# For Jetson Nano: Use TensorFlow 2.13.1 (latest available for Python 3.8 on ARM64)
# Note: TensorFlow GPU requires NVIDIA JetPack which is pre-installed on balena Jetson images
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir "numpy>=1.21.2,<2.0" && \
    pip3 install --no-cache-dir flask==3.0.0 requests==2.32.3 && \
    pip3 install --no-cache-dir tensorflow==2.13.1

COPY . .

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

CMD ["python3", "-u", "app/server.py"]
