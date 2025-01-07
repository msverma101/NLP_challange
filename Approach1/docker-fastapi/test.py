from fastapi.testclient import TestClient
from main import app, TextInput
from fastapi.encoders import jsonable_encoder

client = TestClient(app)

def test_analyze_sentiment():
    test_cases = [
        {"text": "I love this product!", "expected_class_index": 0},  # Positive
        {"text": "This is terrible.", "expected_class_index": 1},      # Negative
        {"text": "This is just okay.", "expected_class_index": 2},     # Neutral
        {"text": "Apples are red.", "expected_class_index": 3},        # Irrelevant
    ]

    for case in test_cases:
        payload = TextInput(text=case["text"])
        response = client.post("/analyze/", json=jsonable_encoder(payload))

        assert response.status_code == 200
        probabilities = response.json()["probabilities"]
        assert len(probabilities) == 4
        predicted_index = probabilities.index(max(probabilities))
        assert predicted_index == case["expected_class_index"]