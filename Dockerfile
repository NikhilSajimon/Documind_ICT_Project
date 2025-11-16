# 1. Base Image
FROM python:3.11-slim

# 2. Set Environment Variables
ENV PYTHONUNBUFFERED True
# Tells gunicorn to listen on the port Cloud Run provides
ENV PORT 8080

# 3. Install System Dependencies (for Tesseract and pdf2image)
# We run this in one layer to reduce image size
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 4. Set up the working directory
WORKDIR /app

# 5. Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy ALL your application code, models, and templates
# This will copy app.py, models/, and templates/
COPY . .

# 7. EXPOSE the port gunicorn will run on
EXPOSE 8080

# 8. Start the production server (Gunicorn)
# This is the correct way to run a Flask app in production.
# It starts 'gunicorn', points it to your 'app.py' file, and finds the
# flask app variable named 'app'.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "300", "app:app"]