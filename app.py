from __future__ import division, print_function
# coding=utf-8
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# Flask utils
# Flask utils
from flask import Flask, request, render_template
from flask import jsonify

# load the xgboost model
filename = 'xgb_model.pkl'
xgBoost = pickle.load(open(filename, 'rb'))

def load_user_rating():
    data = pd.read_csv('user_rating.csv')
    return data

def load_sample_data():
    sample_data = pd.read_csv('final_sample.csv')
    return sample_data

def predict_sentiment(df,product):
    all_text = df['reviews_text']

    word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 3) )
    word_vectorizer.fit(all_text)
    train_word_features = word_vectorizer.transform(all_text)

    char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
    char_vectorizer.fit(all_text)
    train_char_features = char_vectorizer.transform(all_text)
    train_features = hstack([train_char_features, train_word_features])

    xgb_pred=xgBoost.predict(train_features)

    sentiment_df=pd.DataFrame({"product":product,"Sentiment":round((np.count_nonzero(xgb_pred)/xgb_pred.size)*100,2)})

    return sentiment_df

# Define a flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=[ 'POST'])
def upload():

    data = request.get_json()

    userName = data['loginUserName']

    final_data = pd.DataFrame(columns = ['product', 'Sentiment'])
    try:
        data.head()
    except:
        data = load_user_rating()

    if userName not in data['reviews_username'].unique():
        return('Sorry! The username you requested is not in our database. Please check the spelling or try with some other username')
    else:
        product_rating = data.loc[userName].sort_values(ascending=False)[0:20].index.tolist()
        for product in product_rating:
            sample_df = load_sample_data()
            product_df=sample_df[sample_df['name'] == product]
            sentiment = predict_sentiment(product_df,product)
            final_data.append(sentiment)

        result = final_data.sort_values(by='Sentiment', ascending=False).head(5)['product']

    return jsonify(result.tolist())


if __name__ == '__main__':
    print('*** App Started ***')
    app.run(debug=True)

