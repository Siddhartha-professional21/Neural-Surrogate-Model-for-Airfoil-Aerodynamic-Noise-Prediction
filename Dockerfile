
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY features.py .
COPY app.py .
COPY models/ ./models/

# run as a normal user instead of root
RUN useradd -m -u 1000 user
USER user

EXPOSE 8000

# use the port the host asks for, or 8000 when nothing is set
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
