FROM balenalib/jetson-nano-ubuntu:focal

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    python3-pip \
    python3-numpy \
    libusb-1.0-0 \
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
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

CMD ["python3", "-u", "app/server.py"]
