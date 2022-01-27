import streamlit as st
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import Word
import nltk

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

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

def app():
    st.title("TRAIN UNLABELLED DATA")
    st.subheader("Excel file must include:-")
    st.markdown("* title_text: column with article (string)")
    st.markdown("* likes: number of likes (int)")
    st.markdown("* comments: number of comments (int)")
    st.markdown("* shares: number of shares (int)")
    uploaded_file = st.file_uploader("Upload your file into the dataset",type='xlsx')
    if uploaded_file is not None:
        st.success("Successfully uploaded")
        df = pd.read_excel(uploaded_file)
        df = missing_values(df)
        df.info()
        new_dtypes = {
            'title_text': str,
            'likes': int,
            'comments': int,
            'shares':int
        }
        df = df.astype(new_dtypes)
        lower_casing(df)
        punctuation_removal(df)
        stop_word_removal(df)
        rare_words_removal(df)
        common_words_removal(df)
        lemmatization(df)
        x_df_user = df['title_text']
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit_transform(x_df_user)
        count_matrix = tfidf_vectorizer.transform(x_df_user)
        tfidf = TfidfTransformer(norm="l2")
        tfidf.fit(count_matrix)
        tf_idf_matrix = tfidf.fit_transform(count_matrix)
        km = KMeans(n_clusters=2, init='k-means++', max_iter=100, n_init=1, verbose=True, random_state=0)
        # making a pipeline
        km.fit(tf_idf_matrix)
        km.labels_
        df['km_labels'] = km.labels_.tolist()
        fake = df[df.km_labels == 0]
        real = df[df.km_labels == 1]
        st.markdown("Dataset for fake news")
        st.dataframe(fake)
        st.markdown("Dataset for real news")
        st.dataframe(real)
        x = df.drop(columns=['km_labels', 'title_text'])
        y = df['km_labels']
        # over-sampling minority class
        sm = SMOTE()
        x, y = sm.fit_resample(x, y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
        clf = DecisionTreeClassifier(max_depth=5, min_samples_split=2, max_features="auto")
        clf.fit(x_train, y_train)

    else:
        st.warning("Please upload an excel file")
    st.markdown("Custom Fake News Detector:")
    col1, col2, col3 = st.columns(3)
    likes = col1.text_input("Number of likes: ")
    comments = col2.text_input("Number of comments: ")
    shares = col3.text_input("Number of shares: ")
    detect = {
        'comments': comments,
        'shares': shares,
        'likes': likes,
    }
    button_detect = st.button("Detect")
    if button_detect:
        df_detect = pd.DataFrame(detect, index=[0])
        prediction = clf.predict(df_detect)
        if prediction == 1:
            st.success('This is not a fake news')
        if prediction == 0:
            st.warning('This is fake news')
