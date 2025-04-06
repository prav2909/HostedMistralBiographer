# Use a smaller base image
FROM python:3.8-slim

WORKDIR /app

# Copy the requirements file
COPY ./requirements.txt /app/requirements.txt

# Install system-level dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libffi-dev \
    libssl-dev \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN python -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# # Set the environment activation as the default shell command.
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy application files (if any)
# COPY . /app
COPY ./data/ /app/data
COPY ./templates/ /app/templates
COPY ./model.py /app/model.py
COPY ./app_fast.py /app/app_fast.py



# CMD ["fastapi", "run", "app_fast.py", "--port", "8000"]
# Expose the port
EXPOSE 8000

# Run the FastAPI application with uvicorn
CMD ["uvicorn", "app_fast:app", "--host", "0.0.0.0", "--port", "8000"]