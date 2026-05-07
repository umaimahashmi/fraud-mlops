FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create necessary directories
RUN mkdir -p data models mlruns

# Expose port for FastAPI
EXPOSE 8000

# Run the inference API
CMD ["python", "-m", "uvicorn", "src.inference.main:app", "--host", "0.0.0.0", "--port", "8000"]