from flask import Flask, request, jsonify, render_template
# Load data and model logic here
app = Flask(__name__)

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import string
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# Load Data
zomato_real = pd.read_csv('zomato.csv')

# Data Cleaning
zomato = zomato_real.drop(['url', 'dish_liked', 'phone'], axis=1)
zomato.drop_duplicates(inplace=True)
zomato.dropna(how='any', inplace=True)
zomato = zomato.rename(columns={'approx_cost(for two people)': 'cost', 'listed_in(type)': 'type', 'listed_in(city)': 'city'})

# Column transformations
zomato['cost'] = zomato['cost'].apply(lambda x: str(x).replace(',', '')).astype(float)
zomato = zomato.loc[zomato.rate != 'NEW']
zomato = zomato.loc[zomato.rate != '-']
zomato['rate'] = zomato['rate'].apply(lambda x: str(x).replace('/5', '')).astype(float)
zomato['Mean Rating'] = zomato.groupby('name')['rate'].transform('mean')
scaler = MinMaxScaler(feature_range=(1, 5))
zomato[['Mean Rating']] = scaler.fit_transform(zomato[['Mean Rating']])

# Text Preprocessing
lemmatizer = WordNetLemmatizer()
stopwords_set = set(stopwords.words('english'))
PUNCT_TO_REMOVE = string.punctuation

def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))  # Remove punctuation
    text = " ".join([word for word in text.split() if word not in stopwords_set])  # Remove stopwords
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])  # Lemmatize
    return text

zomato['reviews_list'] = zomato['reviews_list'].apply(preprocess_text)

# Recommendation Function
df_percent = zomato.sample(frac=0.5).set_index('name')

# Cosine Similarities
tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_percent['reviews_list'])

# Cuisine vectorization
cuisine_vec = TfidfVectorizer(stop_words='english')
cuisine_matrix = cuisine_vec.fit_transform(df_percent['cuisines'])

# Cost similarity
cost_sim = cosine_similarity(df_percent[['cost']])

# Combined similarity (weighted average)
combined_sim = (0.5 * cosine_similarity(tfidf_matrix)) + (0.3 * cosine_similarity(cuisine_matrix)) + (0.2 * cost_sim)

indices = pd.Series(df_percent.index)

def recommend(name, cosine_similarities=combined_sim):
    idx = indices[indices == name].index[0]
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
    top30_indexes = list(score_series.iloc[0:31].index)

    recommend_restaurant = [list(df_percent.index)[each] for each in top30_indexes]

    df_new = pd.DataFrame(columns=['cuisines', 'Mean Rating', 'cost'])
    for each in recommend_restaurant:
        df_new = df_new.append(df_percent[['cuisines', 'Mean Rating', 'cost']][df_percent.index == each].sample())

    df_new = df_new.drop_duplicates(subset=['cuisines', 'Mean Rating', 'cost'], keep=False)
    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(10)

    return df_new


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_restaurant():
    restaurant_name = request.form['restaurant']
    try:
        recommendations = recommend(restaurant_name)
        return render_template('result.html', tables=[recommendations.to_html(classes='data', header="true")])
    except:
        return "Restaurant not found in the dataset."

if __name__ == "__main__":
    app.run(debug=True)
