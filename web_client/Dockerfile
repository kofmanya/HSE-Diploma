FROM python:3.11

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Expose Flask port
EXPOSE 5050

# Run the Flask app
CMD ["python", "client.py"]

