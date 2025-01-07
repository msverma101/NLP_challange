# Use the official PyTorch image for CPU
FROM pytorch/pytorch:latest

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the application code to the working directory
COPY . /app

# Ensure the model folder is correctly included
COPY ./model ./model

# Ensure the model folder is correctly included
COPY ./data ./data

RUN pip uninstall bitsandbytes
RUN pip install --upgrade transformers


# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application port
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
