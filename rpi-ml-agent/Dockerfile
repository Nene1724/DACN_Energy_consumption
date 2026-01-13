FROM balenalib/raspberry-pi-python:3.10-bullseye-run

RUN apt-get update && apt-get install -y \
    ca-certificates \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

COPY requirements.txt ./
# Install runtimes: 
# Use tensorflow instead of tflite-runtime for FULLY_CONNECTED v12 support
# Install numpy 1.x first, then tensorflow
RUN pip install --no-cache-dir "numpy>=1.21.2,<2.0" && \
    pip install --no-cache-dir flask==3.0.0 requests==2.32.3 && \
    pip install --no-cache-dir tensorflow==2.15.1 \
    pip install pyserial

COPY fnb58_reader.py /app/fnb58_reader.py
COPY . .

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "app/server.py"]
