from joblib import load

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import uvicorn

from process_parallel_text import preprocess_text

tfidf_vectorizer = load("models/tfidf_vectorizer.joblib")
relevance_model = load("models/relevance_model.joblib")
ensemble_model = load("models/ensemble_model.joblib")


# Initialize FastAPI app
app = FastAPI()


# Define a Pydantic model for the input data
class AnalyzeRequest(BaseModel):
    context: str
    text: str


# Welcome endpoint
@app.get("/")
async def welcome():
    # Redirect to the Swagger UI page
    return RedirectResponse(url="/docs")


# Sentiment analysis endpoint
@app.post("/analyze/")
async def predict(request: AnalyzeRequest):
    """
    Predict the sentiment label of a given text within a specific context.
    """
    context = request.context
    text = request.text

    text = context + " " + preprocess_text(text)
    tfidf_text = tfidf_vectorizer.transform([text])
    rel = relevance_model.predict(tfidf_text)
    if rel:
        return {"prediction": "Irrelevant"}

    y_pred = ensemble_model.predict(tfidf_text)[0]
    y_pred_label = (
        "Positive" if y_pred == 1 else "Negative" if y_pred == -1 else "Neutral"
    )
    return {"prediction": y_pred_label}


# Run the FastAPI app using Uvicorn
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
