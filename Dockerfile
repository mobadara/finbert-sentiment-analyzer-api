# Use a lightweight Python base image
FROM python:3.9-slim

# Set up a new user named "user" with user ID 1000
# This is a strict requirement for Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user

# Set environment variables for the user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install them
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY --chown=user . .

# Run Uvicorn on port 7860 (The port HF Spaces listens to)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
