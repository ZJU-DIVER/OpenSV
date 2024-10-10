import sys

sys.path.append("..")

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("./imdb-text.csv")
df['sentiment_id'] = df.sentiment.factorize()[0]
sentiment_id_df = df[['sentiment', 'sentiment_id']].drop_duplicates().sort_values(by = 'sentiment_id').reset_index(drop = 1)
sentiment_to_id = dict(sentiment_id_df.values)

stop_words = stopwords.words('english')
tf_idf = TfidfVectorizer(
    ngram_range=(1, 2), 
    stop_words=stop_words, 
    max_features=100
)

x_train, x_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], train_size = 1000, test_size=200, random_state = 42)

encoder = LabelEncoder()

y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

tf_idf.fit(x_train)

x_train = tf_idf.fit_transform(x_train)
x_test = tf_idf.fit_transform(x_test)

from opensv import DataShapley

shap = DataShapley()
shap.load(x_train, y_train, x_test, y_test)
shap.solve('kernel_shap')
print(shap.get_values())
