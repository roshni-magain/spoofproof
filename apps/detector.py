import streamlit as st
import pandas as pd
import re
from PIL import Image

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import Word
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

ds3 = pd.read_excel("C:/Users/Roshni/Desktop/Shelves/Y3S1/Data Science Project/twitter_fakenews_USElections_2016.xlsx",
                    sheet_name='DATA')
# manipulating and altering the twitter dataset
ds3 = ds3.drop(['tweet_id', 'user_friends_count', 'user_favourites_count', 'tweet_source', 'geo_coordinates_available',
                'num_hashtags', 'num_mentions', 'num_urls', 'user_screen_name', 'fake_news_category_2',
                'is_fake_news_2', 'is_fake_news_1', 'created_at'], axis=1)
ds3 = ds3.rename(columns={'fake_news_category_1': 'Label'})
ds3 = ds3.rename(columns={'text': 'title_text'})
ds3.drop(ds3[ds3['Label'] == 0].index, inplace=True)
ds3.loc[ds3.Label == -1, 'Label'] = 0
ds3.loc[ds3.Label == 1, 'Label'] = 1
ds3.loc[ds3.Label == 2, 'Label'] = 1
ds3.loc[ds3.Label == 3, 'Label'] = 1
ds3.loc[ds3.Label == 4, 'Label'] = 1
ds3.loc[ds3.Label == 5, 'Label'] = 1
ds3 = ds3.sample(frac=1)
ds3 = pd.get_dummies(ds3, columns=['user_verified'])

def missing_values(data):
    data = data.dropna()
    return data


def lower_casing(data):
    data['title_text'] = data['title_text'].apply(lambda x: " ".join(x.lower() for x in x.split()))


def punctuation_removal(data):
    data['title_text'] = data['title_text'].str.replace('[^\w\s]', '',regex=True)


def stop_word_removal(data):
    stop = stopwords.words('english')
    data['title_text'] = data['title_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


def rare_words_removal(data):
    rare = pd.Series(' '.join(data['title_text']).split()).value_counts()[-1000:]
    rare = list(rare.index)
    data['title_text'] = data['title_text'].apply(lambda x: " ".join(x for x in x.split() if x not in rare))


def common_words_removal(data):
    common = pd.Series(' '.join(data['title_text']).split()).value_counts()[:10]
    common = list(common.index)
    data['title_text'] = data['title_text'].apply(lambda x: " ".join(x for x in x.split() if x not in common))


def lemmatization(data):
    lemmatizer = WordNetLemmatizer()
    data['title_text'] = data['title_text'].apply(
        lambda x: " ".join([lemmatizer.lemmatize(Word(word)) for word in x.split()]))

def evaluation(pred_test, y_test):
    print(classification_report(y_test, pred_test))
# DS2
missing_values(ds3)
lower_casing(ds3)
punctuation_removal(ds3)
stop_word_removal(ds3)
rare_words_removal(ds3)
common_words_removal(ds3)
lemmatization(ds3)

x = ds3
tfidf_params = dict(sublinear_tf= True,
                       min_df = 5,
                       norm= 'l2',
                       ngram_range= (1,2),
                       stop_words ='english')
tfidf_vectorizer = TfidfVectorizer(**tfidf_params)



ds3_x_train, ds3_x_test, ds3_y_train, ds3_y_test = train_test_split(x.drop(columns=['Label','user_verified_True']), ds3['Label'],
                                                                    random_state=0, train_size=0.80, test_size=0.15)

column_transformer = ColumnTransformer(
    [('tfidf', tfidf_vectorizer, 'title_text')],
    remainder='passthrough')
clf = DecisionTreeClassifier()
# making a pipeline
pipe = Pipeline([
    ('tfidf', column_transformer),
    ('classify', clf)
])
pipe.fit(ds3_x_train, ds3_y_train)

def app():
    st.title("FAKE NEWS DETECTOR")
    st.markdown("Input the News Content and respective features in the text box below ")
    st.write("Notes:")
    st.write("User Verification  : (0 - verified, 1 - not verified)")
    st.write("Number of followers: Number of followers by the tweet source")
    user_input = st.text_input("Enter news to detect here: ",)
    col1, col2, col3, col4 = st.columns(4)
    user_verified = col1.text_input("User Verification: ")
    retweets = col2.text_input("Number of retweets: ")
    media = col3.text_input("Number of images: ")
    followers = col4.text_input("Number of followers: ")
    detect = st.button("Detect")
    if detect:
        user_input = user_input.lower()
        user_input = re.sub(r'[^\w\s]', '', user_input)
        lemmatizer = WordNetLemmatizer()
        user_input = lemmatizer.lemmatize(user_input)
        data_frame={
            'retweet_count': retweets,
            'title_text': user_input,
            'user_followers_count':followers,
            'num_media': media,
            'user_verified_False':user_verified
        }
        df = pd.DataFrame(data_frame, index=[0])
        prediction = pipe.predict(df)
        if prediction == 0:
            st.success('This is not a fake news')
        if prediction == 1:
            st.warning('This is fake news')
        st.header("About the Model")
        st.subheader("Data Trained")
        st.markdown("Data is extracted from a collection of tweets related to the"
                    "2016 US election between Nov and March of 2018. It is extracted "
                    "using Twitter's API and in specific, about @realDonaldTrump & "
                    "@HillaryClinton")
        st.markdown("Source = https://zenodo.org/record/1048826#.YfFhAOpBzIV")
        st.dataframe(ds3)
        st.subheader("Modeling")
        st.markdown("By using scikit learn's Decision Tree machine learning algorithm")
        st.markdown("Resource : https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html")

        st.subheader("How does a model without features differ from a model with features?")
        st.markdown("Through experimentation, the accuracy/f1-scores of a model trained"
                    "with features and without have differences as shown below:- ")
        img = Image.open("f1_score.png")
        st.image(img)
        st.markdown("It can be seen that the model with features had a higher f1-score on both the labels compared to the model with features")
        st.subheader("Words associated to ""Fake News"" ")
        img2=Image.open("wordcloud.png")
        st.image(img2)
