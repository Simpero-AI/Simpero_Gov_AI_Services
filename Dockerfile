FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for docling (opencv and image processing libs)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install fastapi, uvicorn, and docling (with CPU-only PyTorch)
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir docling --extra-index-url https://download.pytorch.org/whl/cpu

# Set environment variables for better performance
ENV OMP_NUM_THREADS=4
ENV HF_HOME=/tmp/
ENV TORCH_HOME=/tmp/

COPY parser_service/ /app/parser_service/

EXPOSE 8001

CMD ["uvicorn", "parser_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
