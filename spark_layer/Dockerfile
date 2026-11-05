FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PYSPARK_PYTHON=python3 \
    PYSPARK_DRIVER_PYTHON=python3

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        procps \
        curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY spark_layer/requirements.txt /app/spark_layer/requirements.txt
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 \
    -r /app/spark_layer/requirements.txt

COPY spark_layer/ /app/spark_layer/

CMD ["python", "spark_layer/stream_processor.py"]
