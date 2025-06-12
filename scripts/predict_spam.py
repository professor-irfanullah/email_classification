import pickle
with open('models/high_accuracy_classification_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

new_email = input('Enter your email to predict if spam or ham:')
new_emial_vec = model_data['vectorizer'].transform([new_email])
prediction = model_data['model'].predict(new_emial_vec)
print(f"Prediction:{prediction[0]}\n")
print('accuracy:',model_data['metadata']['accuracy'])
print('classification_report:\n',model_data['metadata']['classification_report'])
