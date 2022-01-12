# Imports

import numpy as np
import pandas as pd
import streamlit as st
import re
import sklearn

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

st.title('MovieRecs')
st.markdown('By [Jen Zieger](https://github.com/jenzieg)')

st.markdown('Grab some popcorn! \nYou\'re about to watch a movie you\'re actually in the mood for!')
st.markdown('\n\n\nPlease use the below form to input what you think you\'d want to watch.\nThis can be a title, an actor, a genre, a mood, you name it! \n\n\nThe more detailed you are, the better the recommendations will be.\n\n\nSome examples include:'
          '\n* I want to watch a scary horror movie'
          '\n * Scorsese'
          '\n * College or High School Comedy'
          '\n * I\'m in the mood for a gut-wrenching dramatic movie'
          '\n * I want to watch a movie with Jordan Peele in it'
          '\n * International'
          '\n * Meryl Streep Comedy')

df = pd.read_csv('./assets/df_for_flask.csv')
tokenizer = word_tokenize
stop_words = stopwords
stemmer = WordNetLemmatizer()

english = stopwords.words('english')
english = set(english)
my_stop_words = ['maybe','like','kind of','sort of','similar to', 'want', 'watch','content',
                'critics','review','movie','lacks','perfectly acceptable', 'consensus','best',
                 'excellent', "'d", "'ll", "'re", "'s", 'want','watch','I want to watch',
                 "'ve", 'could', 'might', 'must', "n't", 'need', 'sha', 'wo','actor','director','producer',
                 'would', 'acceptable', 'kind', 'perfectly', 'similar', 'sort','i','of','to']
my_stop_words = set(my_stop_words)
new_stop_words = english.union(my_stop_words)

# Get best index
def get_idx(matrix):

    # returns sum of all tokens cosines for each sentence
    cos_sim = np.mean(matrix, axis=0)

    # Ranking index from highest to smallest
    index = np.argsort(cos_sim)[::-1]

    # Returning an array of cosine similarity shape, filled with 1s
    mask = np.ones(len(cos_sim))

    # Setting up truth value to ensure arrays are same shape
    mask = np.logical_or(cos_sim[index], mask)

    # Assigning index
    best_idx = index[mask][:35]

    return best_idx


# Get recomendations
def recs_tfidf(text, tfidf_matrix):

    # Removing contractions
    # text = contractions.fix(text)
    letters_only = re.sub("[^a-zA-Z]",  " ", str(text))
    # Get tokens
    tokens = [str(t) for t in tokenizer(letters_only)]

    # Vectorize text
    text = vectorizer.transform(tokens)

    # Create list with similarity between text and dataset
    matrix = cosine_similarity(text, tfidf_matrix)

    best_idx = get_idx(matrix)

    return best_idx

def get_final(text, new_rec):
    if 'drama' in text.lower():
        mask = new_rec['genre'].str.contains('Drama' or 'Romance', case=False, na=False)
        new_rec = new_rec[mask]
        new_rec = new_rec.sort_values(by = ['score'], ascending = False)[:20]
    elif 'comedy' in text.lower() or 'slapstick' in text.lower():
        mask = new_rec['genre'].str.contains('Comedy', case=False, na=False)
        new_rec = new_rec[mask]
        new_rec = new_rec.sort_values(by = ['score'], ascending = False)[:20]
    elif 'horror' in text.lower() or 'scary' in text.lower():
        mask = new_rec['genre'].str.contains('Horror', case=False, na=False)
        new_rec = new_rec[mask]
        new_rec = new_rec.sort_values(by = ['score'], ascending = False)[:20]
    elif 'action' in text.lower() or 'adventure' in text.lower():
        mask = new_rec['genre'].str.contains('Action & Adventure', case=False, na=False)
        new_rec = new_rec[mask]
        new_rec = new_rec.sort_values(by = ['score'], ascending = False)[:20]
    elif 'mystery' in text.lower() or 'suspense' in text.lower():
        mask = new_rec['genre'].str.contains('Mystery & Suspense', case=False, na=False)
        new_rec = new_rec[mask]
        new_rec = new_rec.sort_values(by = ['score'], ascending = False)[:20]
    elif 'international' in text.lower() or 'art' in text.lower():
        mask = new_rec['genre'].str.contains('Art House & International', case=False, na=False)
        new_rec = new_rec[mask]
        new_rec = new_rec.sort_values(by = ['score'], ascending = False)[:20]
    elif 'romantic' in text.lower() or 'romance' in text.lower():
        mask = new_rec['genre'].str.contains('Romance', case=False, na=False)
        new_rec = new_rec[mask]
        new_rec = new_rec.sort_values(by = ['score'], ascending = False)[:20]
    elif 'classic' in text.lower():
        mask = new_rec['genre'].str.contains('Classic', case=False, na=False)
        new_rec = new_rec[mask]
        new_rec = new_rec.sort_values(by = ['score'], ascending = False)[:20]
    else:
        new_rec
    return new_rec

def get_recs(text):
    best_idx = recs_tfidf(text, tfidf_matrix)
    new_rec = (df[['title','plot','genre','score','tomatometer_rating','audience_rating','imdb_score']].iloc[best_idx])
    new_rec = new_rec.sort_values(by = ['score'], ascending = False)
    new_rec.drop_duplicates(subset = 'title', keep = 'first', inplace = True)
    rec = get_final(text,new_rec)
    rec.drop_duplicates(subset = 'title', keep='first',inplace = True)
    rec = rec[:10]
    rec = rec.reset_index()
    rec = rec.drop(columns = 'index')
    return rec

vectorizer = TfidfVectorizer(stop_words = new_stop_words, lowercase = True, tokenizer = tokenizer, ngram_range = (1,4))
tfidf_matrix = vectorizer.fit_transform(df['text'].values)

key_word = st.text_input('Input what you\'re in the mood for here:')

if st.button('Search 🔍'):
	st.dataframe(get_recs(key_word))
