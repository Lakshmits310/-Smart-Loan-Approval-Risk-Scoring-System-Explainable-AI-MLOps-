# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files (dashboard + models)
COPY dashboard/ dashboard/
COPY models/ models/

# Expose port for Streamlit
EXPOSE 8501

# Command to run app
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
