import re
import spacy

from Bio.Align import PairwiseAligner, MultipleSeqAlignment
from Bio import pairwise2

from collections import Counter




# Load spaCy model for tokenization and lemmatization
nlp = spacy.load("en_core_web_sm")   # python -m spacy download en_core_web_sm

def preprocess_text(text):
    """
    Preprocess a string by removing URLs, non-alphabetic characters, extra spaces, and converting to lowercase.
    """
    # Remove URLs (http, https, www)
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove any links with format word.ext/word
    text = re.sub(r'\w+\.\w+/\w+', '', text)
    
    # Remove any non-alphabetic characters, keeping spaces (including strange symbols, emojis)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove filler words like "uhh"
    text = re.sub(r'\b(uh+|um+|ah+|er+)\b', '', text)
    
    # Normalize elongated words (e.g., "gooooo" -> "go")
    text = re.sub(r'(\w)\1{2,}', r'\1', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Convert to lowercase
    text = text.lower() 
    
    # # Tokenize and lemmatize using spaCy, removing stopwords
    # doc = nlp(text)
    # tokens = [token.lemma_ for token in doc if token.text not in stopwords.words('english') and token.is_alpha]
    
    # # Join the tokens back into a cleaned string
    # text = ' '.join(tokens)
    
    return text

def align_comments(comments):
    """Align all comments to the longest one."""
    reference = max(comments, key=len).split()
    aligned_comments = []

    for comment in comments:
        comment = comment.split()
        alignment = pairwise2.align.globalxx(
            reference, comment, one_alignment_only=True, gap_char=["-"]
        )
        aligned_comments.append(alignment[0][1])

    return aligned_comments


def majority_vote(aligned_comments):
    """Perform majority voting on aligned comments."""
    # Ensure all tokenized lists have the same length by padding
    max_len = max(map(len, aligned_comments))
    padded_tokens = [
        tokens + [""] * (max_len - len(tokens)) for tokens in aligned_comments
    ]

    # Perform majority voting at each position
    final_tokens = []
    for i in range(max_len):
        words = [tokens[i] for tokens in padded_tokens]
        most_common_word = Counter(words).most_common(1)[0][0]
        final_tokens.append(most_common_word)

    return " ".join(final_tokens).replace("-", "").strip()


merge_comments = lambda comments: majority_vote(align_comments(comments))


if __name__ == "__main__":
    # Input comments
    comments = [
        "Happy Monday Everyone! Another Happy day in Animal Crossing Followed by assassins creed! We're going Live! HOP IN!",
        "Happy Monday Everyone! Another Happy Day in Animal Crossing Followed by the Creed of the Assassins! We're going Live! HOP IN!",
        "Happy Monday! Another Happy Day at Animal! Sing together! We're going to Live! Hope!",
        "Happy Monday Everyone! Another Happy day in Animal Crossing Followed by assassins creed! We're going Live! HOP IN!",
        "to Happy Monday Everyone! Another Happy day in Animal Crossing Followed by assassins creed! We're going Live! HOP IN!",
    ]
    final_comment = merge_comments(comments)

    print(f"Final comment: {final_comment}")
