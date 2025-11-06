# Dockerfile for AI Dating App AI Service
# Optimized for Render deployment

FROM python:3.12.2

# Set working directory
WORKDIR /app

# Install system dependencies needed for ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories for embeddings and data
RUN mkdir -p ai/ai_date_planner/embeddings ai/ai_lovabot/data

# Expose port (Render will set PORT environment variable)
EXPOSE 8000

# Suppress warnings from sentence-transformers
ENV PYTHONWARNINGS="ignore::SyntaxWarning:sentence_transformers.*"

# Start the FastAPI server
# Render sets PORT automatically, default to 8000 if not set
# Use shell form to allow environment variable expansion
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}

