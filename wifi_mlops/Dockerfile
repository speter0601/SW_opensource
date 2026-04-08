# Use a lightweight official Python 3.10 slim image
FROM python:3.10-slim

# Prevent Python from writing .pyc files & ensuring output is logged directly
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements first to leverage Docker cache for dependencies
COPY requirements.txt .

# Install dependencies without keeping the cache to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of our application code (including pre-trained artifacts if available)
COPY . .

# Expose the API port
EXPOSE 8000

# Command to run our FastAPI server automatically
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
