FROM balenalib/beaglebone-black-debian-python:3.9-bullseye-run

RUN apt-get update && apt-get install -y \
    ca-certificates \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir --no-deps tflite-runtime==2.14.0 && \
    pip install --no-cache-dir flask==3.0.0 requests==2.32.3

COPY . .

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "app/server.py"]
