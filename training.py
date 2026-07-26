import json
import numpy as np
import random
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

with open('intents.json') as file:
    data = json.load(file)


words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']


for intent in data['intents']:
    for pattern in intent['patterns']:
        
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_words]
words = sorted(list(set(words)))


classes = sorted(list(set(classes)))


pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))


training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])


random.shuffle(training)
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])


model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))


adam = Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)


model.save('model.h5')
print("Model training complete. Files saved: model.h5, labels.pkl, texts.pkl")
