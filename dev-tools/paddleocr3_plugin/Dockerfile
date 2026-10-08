FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PADDLEX_HOME=/opt/paddlex \
    PADDLE_PDX_CACHE_HOME=/opt/paddlex

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY dev-tools/paddleocr3_plugin/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir paddlepaddle==3.3.1 \
    && python -m pip install --no-cache-dir -r /app/requirements.txt

COPY dev-tools/paddleocr3_plugin/worker.py /app/worker.py
VOLUME ["/opt/paddlex"]
ENTRYPOINT ["python", "/app/worker.py"]
CMD ["--ocr-version", "PP-OCRv6", "--cpu-threads", "4"]
