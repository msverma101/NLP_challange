import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from joblib import dump

# Load and preprocess data
train_preprocessed_data = pd.read_csv('data/training_preprocessed.csv').dropna()

eval_preprocessed_data = pd.read_csv('data/validation_preprocessed.csv').dropna()

# Step 1: Relevance Classification
X_train_rel = train_preprocessed_data['Context'].str.cat(train_preprocessed_data['FinalComment'], sep=' ')
y_train_rel = train_preprocessed_data['Label']=="Irrelevant"

X_test_rel = eval_preprocessed_data['Context'].str.cat(eval_preprocessed_data['FinalComment'], sep=' ')
y_test_rel = eval_preprocessed_data['Label']=="Irrelevant"

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_rel_tfidf = tfidf_vectorizer.fit_transform(X_train_rel.to_list())
X_test_rel_tfidf = tfidf_vectorizer.transform(X_test_rel.to_list())

relevance_model = RandomForestClassifier(random_state=42)
relevance_model.fit(X_train_rel_tfidf, y_train_rel)
y_pred_rel = relevance_model.predict(X_test_rel_tfidf)
report = classification_report(y_test_rel, y_pred_rel)
print("Relevance Classification Report:")
print(report)
with open('results/classification_report.txt', 'w') as file:
    file.write("Relevance Classification Report\n")
    file.write(report)
    file.write("\n")
    
# Save relevance models
dump(tfidf_vectorizer, 'models/tfidf_vectorizer.joblib')
dump(relevance_model, 'models/relevance_model.joblib')


# Step 2: Sentiment Analysis on Relevant Data
label_mapping = {"Positive": 1, "Negative": -1, "Neutral": 0}
X_train_sent = X_train_rel[~y_train_rel]
y_train_sent = train_preprocessed_data['Label'][~y_train_rel]
y_train_sent = y_train_sent.map(label_mapping)

X_test_sent = X_test_rel[~y_test_rel]
y_test_sent = eval_preprocessed_data['Label'][~y_test_rel]
y_test_sent = y_test_sent.map(label_mapping)

X_train_sent_tfidf = tfidf_vectorizer.transform(X_train_sent.to_list())
X_test_sent_tfidf = tfidf_vectorizer.transform(X_test_sent.to_list())

# Train individual sentiment models
svm_model = SVC(probability=True, random_state=42)
log_reg_model = LogisticRegression(random_state=42)
rf_model = RandomForestClassifier(random_state=42)

svm_model.fit(X_train_sent_tfidf, y_train_sent)
log_reg_model.fit(X_train_sent_tfidf, y_train_sent)
rf_model.fit(X_train_sent_tfidf, y_train_sent)

# Ensemble with majority voting
ensemble_model = VotingClassifier(estimators=[
    ('svm', svm_model),
    ('logreg', log_reg_model),
    ('rf', rf_model)
], voting='soft')
ensemble_model.fit(X_train_sent_tfidf, y_train_sent)

y_pred_sent = ensemble_model.predict(X_test_sent_tfidf)

report = classification_report(y_test_sent, y_pred_sent, target_names=label_mapping.keys())
print("Emsemble Classification Report:")
print(report)

with open('results/classification_report.txt', 'a') as file:
    file.write("Sentiment Analysis Classification Report of Ensemble Model\n")
    file.write(report)

dump(ensemble_model, 'models/ensemble_model.joblib')


