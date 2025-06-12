import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score , classification_report
from sklearn.model_selection import cross_val_score

df = pd.read_csv('data/spam_ham_dataset.csv')

X = df['text']
y = df['label']
X_train , X_test , y_train , y_test = train_test_split(X,y , test_size = 0.5 , random_state = 42)

vectorizer = CountVectorizer(stop_words='english', lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec , y_train)
predictions = model.predict(X_test_vec)

new_email = input('Enter your email to predict if spam or ham:')
new_email_vec = vectorizer.transform([new_email])
prediction = model.predict(new_email_vec)
accuracy = accuracy_score( y_test,predictions)
classification = classification_report( y_test,predictions )
scores = cross_val_score(model, X_train_vec, y_train, cv=5)

print(f"Prediction_result:{prediction[0]}") 
print('\n Accuracy:' , accuracy)
print('classification_report:\n' , classification)
print("Cross-validation accuracy:", scores.mean())
