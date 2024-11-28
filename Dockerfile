FROM openjdk:11-jdk-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    python3-dev \
    wget \
    procps

RUN wget https://archive.apache.org/dist/spark/spark-3.3.4/spark-3.3.4-bin-hadoop3.tgz \
    && tar -xzf spark-3.3.4-bin-hadoop3.tgz \
    && mv spark-3.3.4-bin-hadoop3 /opt/spark \
    && rm spark-3.3.4-bin-hadoop3.tgz

ENV JAVA_HOME=/usr/local/openjdk-11
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin
ENV PYTHONPATH=/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.5-src.zip

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY prediction.py .
COPY TestDataset.csv /app/TestDataset.csv

RUN mkdir -p /app/output

ENTRYPOINT ["spark-submit", "prediction.py"]
CMD ["/app/TestDataset.csv"]  # Default path if none provided
