# Start with Python 3.8 base image
FROM python:3.8-slim

# Install system essentials
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a Python virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install MLflow and other dependencies with specific versions
RUN pip install --no-cache-dir \
    mlflow==2.0.1 \
    scikit-learn==1.1.3 \
    pandas \
    numpy

# Copy your model files
COPY . /opt/ml/model

# Set working directory
WORKDIR /opt/ml/model

# Use the full path to mlflow executable
CMD ["python", "-m", "mlflow", "models", "serve", "-h", "0.0.0.0", "-m", "model", "--env-manager=local"]
