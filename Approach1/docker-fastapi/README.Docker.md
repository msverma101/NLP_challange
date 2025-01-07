# Tweet Sentiment Classification API

This project provides a containerized sentiment analysis API built with FastAPI and Hugging Face Transformers. It classifies the sentiment of tweets into four categories: positive, negative, neutral, and irrelevant.

## Project Structure

```
tweet-sentiment-analysis/
├── main.py       (FastAPI application)
├── model/
│   └── model.pth     (Your trained model weights)
├── .dockerignore    (Files to exclude from the Docker image)
├── Dockerfile        (Instructions for building the Docker image)
├── requirements.txt  (Project dependencies)
├── pyproject..toml  (Project dependencies)
├── test.py           (Unit tests for the API)
└── docker-compose.yml (Optional: For managing with Docker Compose)
```

## Building and Running Locally

**Single Container (Recommended for simple setups):**

1.  **Build the Docker image:**

    ```bash
    docker build -t sentiment-analysis-api .
    ```

2.  **Run the Docker container:**

    ```bash
    docker run -p 8000:8000 sentiment-analysis-api
    ```

**Using Docker Compose (Optional, useful for multi-container apps):**

1.  **Create `docker-compose.yml` (if you don't have one):**

    ```yaml
    version: "3.9"

    services:
      web:
        build:
          context: .
        ports:
          - "8000:8000"
        restart: unless-stopped
    ```

2.  **Build and run using Docker Compose:**

    ```bash
    docker compose up --build
    ```

3.  **Stop using Docker Compose:**

    ```bash
    docker compose down
    ```

**Accessing the API:**

Once the container is running (using either method), the API will be available at http://localhost:8000. You can test it using tools like `curl` or Postman, or by running the provided unit tests (`test.py`).

## Pushing to a Container Registry

To deploy your API to a cloud platform or share it, you'll need to push the Docker image to a container registry (e.g., Docker Hub, Google Container Registry, Amazon ECR).

1.  **Build the image (replace `<your_registry_username>`):**

    ```bash
    docker build -t <your_registry_username>/sentiment-analysis-api .
    ```

2.  **Login to your registry (if needed):**

    ```bash
    docker login <your_registry_url>  # e.g., docker login docker.io
    ```

3.  **Push the image:**

    ```bash
    docker push <your_registry_username>/sentiment-analysis-api
    ```

## Running the Tests

To run the unit tests, you'll typically do this *before* building the Docker image. Make sure you have the required dependencies installed in your local Python environment (you can use a virtual environment). Then, run:

```bash
python -m pytest test.py