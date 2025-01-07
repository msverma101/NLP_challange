from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# VADER Integration
vader_analyzer = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    scores = vader_analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'positive'
    elif scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Combine VADER with ensemble predictions
def combined_sentiment_prediction(context, text):
    text = context + ' ' + preprocess_text(text)
    tfidf_text = tfidf_vectorizer.transform([text])
    rel = relevance_model.predict(tfidf_text)
    if not rel:
        return "Irrelevant"
    
    vader_pred = vader_sentiment(text)
    ensemble_pred = ensemble_model.predict(tfidf_text)[0]
    # Combine or prioritize based on validation performance
    return vader_pred if vader_pred != 'neutral' else ensemble_pred