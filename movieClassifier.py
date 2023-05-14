import sklearn as sk
import nltk
import pandas as pd 
from sklearn.datasets import load_files
from nltk.stem import WordNetLemmatizer



#loading the data
movie_data = load_files("/Users/anasmohamadsobhi/Downloads/review_polarity/txt_sentoken")
X, y = movie_data.data, movie_data.target  # "neg" and "pos" folders into X while y are the targets

print(movie_data.target)
#preprocessing the text

from nltk.stem import WordNetLemmatizer
import re



documents = []
stemmer = WordNetLemmatizer()


for sen in range(0, len(X)):
    # Removing all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))
      
    # removing all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # removing single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # converting to Lowercase
    document = document.lower()
    # removing word ---> the
    document = re.sub(r'the', ' ', document )
    
    # lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(use_idf = True)
X = tfidf_vectorizer.fit_transform(documents).toarray()
#split the the data to training data and testing data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# here split the data to 20% test data and 80% training data


from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression


Model_classifier = LogisticRegression()

#classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
Model_classifier.fit(X_train, y_train) 
y_pred = Model_classifier.predict(X_test)
print(y_pred)