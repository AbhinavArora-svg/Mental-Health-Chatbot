from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import json

app = Flask(__name__)


nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()


try:
    model = tf.keras.models.load_model('model.h5')
    with open('intents.json') as file:
        intents = json.load(file)
    words = pickle.load(open('texts.pkl', 'rb'))
    classes = pickle.load(open('labels.pkl', 'rb'))
except Exception as e:
    print(f"Error loading files: {str(e)}")

def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bag_of_words(sentence):
    sentence_words = clean_sentence(sentence)
    bag = [1 if word in sentence_words else 0 for word in words]
    return np.array(bag)

def predict_intent(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > 0.25]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": float(r[1])} for r in results]

def get_response(intents_list):
    if not intents_list:
        return "I'm not sure how to respond to that. Can you rephrase?"
    tag = intents_list[0]['intent']
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return np.random.choice(intent['responses'])
    return "I'm still learning. Could you tell me more?"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['GET'])
def get_bot_response():
    user_message = request.args.get('msg')
    if not user_message or user_message.strip() == "":
        return jsonify({"response": "Hello! How can I help you today?"})
    
    try:
        intents_list = predict_intent(user_message)
        bot_response = get_response(intents_list)
        return jsonify({"response": bot_response})
    except Exception as e:
        print(f"Error processing message: {str(e)}")
        return jsonify({"response": "I'm having trouble understanding. Could you try again?"})

if __name__ == '__main__':
    app.run(debug=True)
