FROM python:3.9-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Auth module is mounted at /app/auth by docker-compose
ENV PYTHONPATH="/app:/app/auth:${PYTHONPATH}"

# Create shared data directory
RUN mkdir -p /app/shared_data

# Expose port
EXPOSE 8000

# Start the application with debugging
CMD ["python", "-u", "app.py"] 