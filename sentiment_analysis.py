import pandas as pd

import nltk
nltk.download('stopwords')  # This ensures the resource is downloaded before use

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st


stop_words = set(stopwords.words('english'))

df = pd.read_csv("Reviews.csv")  # nrows=5000 - minimize time of execution
df.head()

df = df[['Text', 'Score']].dropna()
#print(df.head())   # first 5 rows of your data
print(df.info())   #  summary info: columns, data types, missing values
#print(df)
#print(df.to_string())
#print(pd.options.display.max_rows)


# Droping missing or unnecessary columns
df = df[['Text','Score']].dropna()

# Maping scores to sentiment labels
def sentiment(score):
    if score <=2:
        return 'negative'
    elif score ==3:
        return 'neutral'
    else:
        return 'positive' 
    
df['Sentiment'] = df['Score'].apply(sentiment)

#Droping neutral for binary classification
df = df[df['Sentiment'] != 'neutral']
print(df.head())


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-letters
    text = text.lower().split()  # Lowercase and tokenize
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

df['Cleaned_Text'] = df['Text'].apply(preprocess)
#print(df[['Text', 'Cleaned_Text']].head())

sns.countplot(x='Sentiment', data=df)
print("Sentiment graph")
plt.title("Sentiment Distribution")
plt.show()

# Filter data
positive_df = df[df['Sentiment'] == 'positive']
negative_df = df[df['Sentiment'] == 'negative']

# Undersample the larger class
min_len = min(len(positive_df), len(negative_df))
balanced_df = pd.concat([positive_df.sample(min_len), negative_df.sample(min_len)])

# Shuffle the dataset
df = balanced_df.sample(frac=1, random_state=42)



#####################################################
print("WordCloud")
positive_words = ' '.join(df[df['Sentiment']=='positive']['Cleaned_Text'])
negative_words = ' '.join(df[df['Sentiment']=='negative']['Cleaned_Text'])



plt.imshow(WordCloud(width=800, height=400).generate(positive_words), interpolation='bilinear')
plt.axis('off')
plt.title("Positive Reviews Word Cloud")
plt.show()


plt.imshow(WordCloud(width=800, height=400).generate(negative_words), interpolation='bilinear')
plt.axis('off')
plt.title("Negative Reviews Word Cloud")
plt.show()
print("\n")
#####################################################
print("vectorizer/ Term Frequency-Inverse Document Frequency(TF-IDF)")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Cleaned_Text'])
y = df['Sentiment']

print("Shape of X (documents x features):", X.shape)
print("First 5 feature names:", vectorizer.get_feature_names_out()[:5])

# the TF-IDF vector for the first document:
print("TF-IDF vector for first document:\n", X[0].toarray())

# Printing the first 10 sentiment labels
print("First 10 labels:", y[:10].tolist())
#####################################################
print("\n")
print("Train-Test Split")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Print shapes to verify the split
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Length of y_train:", len(y_train))
print("Length of y_test:", len(y_test))

# Optional: preview some labels
print("\nFirst 5 training labels:", y_train[:5].tolist())
print("First 5 testing labels:", y_test[:5].tolist())


print("\n")
##############################################
print(df['Sentiment'].value_counts())

print (" Training a Classifier")
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\n")
##############################################

print (" User Input")
def predict_sentiment(text):
    print ("1")
    cleaned = preprocess(text)
    print ("2")
    vect = vectorizer.transform([cleaned])
    print ("3")
    return model.predict(vect)[0]
    print ("4")

print(predict_sentiment("This product is amazing! I love it."))

print("\n")
##############################################

# -------------------------------
# STEP 9: (NEW) Save model + vectorizer
# -------------------------------
import pickle                                                
with open("model.pkl", "wb") as f:                          
    pickle.dump(model, f)                                    

with open("vectorizer.pkl", "wb") as f:                      
    pickle.dump(vectorizer, f)                               

