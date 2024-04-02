# %%
# !pip install -U sentence-transformers

# %%
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')
VERB_CODES = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}

# %%
# Load your dataset
data_path = 'DataNeuron_Text_Similarity.csv'  # Adjust to your dataset path
data = pd.read_csv(data_path)

# %%
def preprocess_text(text):
    text = text.lower()
    temp_sent =[]
    words = nltk.word_tokenize(text)
    tags = nltk.pos_tag(words)
    for i, word in enumerate(words):
        if tags[i][1] in VERB_CODES:
            lemmatized = lemmatizer.lemmatize(word, 'v')
        else:
            lemmatized = lemmatizer.lemmatize(word)
        if lemmatized not in stop_words and lemmatized.isalpha():
            temp_sent.append(lemmatized)

    finalsent = ' '.join(temp_sent)

    return finalsent

# %%
# Apply preprocessing to your dataset (for TF-IDF)
data['text1'] = data['text1'].apply(preprocess_text)
data['text2'] = data['text2'].apply(preprocess_text)

# %%
# Assume your dataset has two columns, 'text1' and 'text2', for each pair of texts
texts1 = data['text1'].tolist()
texts2 = data['text2'].tolist()

# %%
# Initialize the Sentence Transformer model
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Compute embeddings for each text
embeddings1 = model.encode(texts1, batch_size=32, show_progress_bar=True)
embeddings2 = model.encode(texts2, batch_size=32, show_progress_bar=True)

# %%
# Compute cosine similarities between each pair of texts
similarities = [cosine_similarity([emb1], [emb2])[0][0] for emb1, emb2 in zip(embeddings1, embeddings2)]

# Create a DataFrame to store results
results_df = pd.DataFrame({
    'text1': texts1,
    'text2': texts2,
    'similarity': similarities
})

# %%
results_df.head()

# %%
results_df.to_csv('similarity_results.csv', index=False)

# %%
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame with the similarity scores
plt.figure(figsize=(8, 4))
plt.hist(results_df['similarity'], bins=30, color='blue', edgecolor='black')
plt.title('Distribution of Similarity Scores')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


