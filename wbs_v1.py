import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
# nltk.download('stopwords')
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
sns.set()
from nltk.tokenize import sent_tokenize, word_tokenize

csv.register_dialect('mydialect', csv.QUOTE_ALL)

filename = r"C:\Users\Kiril\Downloads\sentiment-analysis-on-financial-tweets\stockerbot-export1.csv"
dataset = pd.read_csv(r"C:\Users\Kiril\Downloads\sentiment-analysis-on-financial-tweets\stockerbot-export1.csv")
# counter = 0
# with open(filename, "r+", encoding="UTF-8", newline="") as file:
#     reader = csv.reader(file, dialect='mydialect')
#     for row in reader:
#         sents = sent_tokenize(row[1])
#         print(sents)
#         # print(row)
#         counter += 1
#     # print(len(list(reader)))
#
#
# print(counter)


dataset = dataset.drop('id',axis=1)
dataset['url'] = dataset['url'].fillna('http://www.NULL.com')
# print(dataset.isnull().sum())
# plt.figure(figsize=(15,6))
# dataset['source'].value_counts()[:10].plot(kind='barh',color=sns.color_palette('summer',30))
# plt.title('Source with most number of tweets')
# plt.show()


pat1 = r'@[A-Za-z0-9]+' # this is to remove any text with @....
pat2 = r'https?://[A-Za-z0-9./]+'  # this is to remove the urls
combined_pat = r'|'.join((pat1, pat2))
pat3 = r'[^a-zA-Z]' # to remove every other character except a-z & A-Z
combined_pat2 = r'|'.join((combined_pat,pat3)) # we combine pat1, pat2 and pat3 to pass it in the cleaning steps

ps = PorterStemmer()
cleaned_tweets = []

for i in range(0, len(dataset['text'])) :
    tweets = re.sub(combined_pat2,' ',dataset['text'][i])
    tweets = tweets.lower()
    tweets = tweets.split()
    tweets = [ps.stem(word) for word in tweets if not word in set(stopwords.words('english'))]
    tweets = ' '.join(tweets)
    cleaned_tweets.append(tweets)

# print(cleaned_tweets[:10])
# print(dataset.columns)
dataset['cleaned_tweets'] = cleaned_tweets

sia = SentimentIntensityAnalyzer()
for tweet in cleaned_tweets[:10]:
    # print(tweet)
    s = sia.polarity_scores(tweet)
    # for k in sorted(s):
    #     print('{0}: {1}, '.format(k, s[k]), end='')
    #     print()


def findpolarity(data):
    sid = SentimentIntensityAnalyzer()
    polarity = sid.polarity_scores(data)
    if(polarity['compound'] >= 0.2):
        sentiment = 1
    if(polarity['compound'] <= -0.2):
        sentiment = -1
    if(polarity['compound'] < 0.2 and polarity['compound'] >-0.2):
        sentiment = 0
    return(sentiment)

sentiment = []
for i in range(0, len(cleaned_tweets)):
    s = findpolarity(cleaned_tweets[i])
    sentiment.append(s)


tweet_sentiment = pd.DataFrame()
tweet_sentiment['cleaned_tweets'] = cleaned_tweets
tweet_sentiment['sentiment'] = sentiment
# tweet_sentiment.to_csv('tweet_sentiment.csv', index=False)



cv = CountVectorizer()
X = cv.fit_transform(tweet_sentiment['cleaned_tweets']).toarray()
y = tweet_sentiment['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

# print(X, y)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)

print(cm)
print(score)