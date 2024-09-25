# AI-POWERED-CHATBOT-WITH-GUI
This chatbot project utilizes Natural Language Processing (NLP) and machine learning to understand and respond to user inputs in real-time. Built with a neural network model, it classifies user queries into predefined intents using tokenization, lemmatization, and a bag-of-words approach.

### 1. **Natural Language Toolkit (NLTK) and Lemmatization**
   - This project uses `nltk` and its `WordNetLemmatizer` to preprocess text data by breaking down user inputs into tokens and converting them to their base form, improving consistency.

### 2. **Model Loading and Preprocessing**
   - A pre-trained model (`chatbot_model.h5`) is loaded, along with tokenized words and classes from `words.pkl` and `classes.pkl`, enabling the chatbot to map inputs to specific intents.

### 3. **Sentence Cleanup**
   - The `clean_up_sentence` function tokenizes and lemmatizes user input, converting each word to lowercase to ensure accurate predictions during processing.

### 4. **Bag of Words (BoW)**
   - `bag_of_words` creates a numerical representation of user input by comparing words with the known vocabulary. This step is key for feeding input into the model.

### 5. **Predicting Class of Input**
   - The `predict_class` function processes the input through the trained model and predicts possible intents, filtering results based on a predefined error threshold.

### 6. **Generating Chatbot Responses**
   - The `getResponse` function takes predicted intents and randomly selects a corresponding response from a set of predefined replies for that intent.

### 7. **Intent Recognition and Data Preparation**
   - In `train_chatbot.py`, the chatbot’s intents are extracted from `intents.json`, tokenized, and categorized into patterns and tags to serve as input-output pairs for training.

### 8. **Training Data Creation**
   - Tokenized patterns are used to generate feature vectors, while the intents are transformed into one-hot encoded vectors, making them suitable for training the model.

### 9. **Model Architecture**
   - The chatbot uses a Sequential Neural Network with three layers: 
     - An input layer with 128 neurons,
     - A hidden layer with 64 neurons,
     - An output layer equal to the number of intent classes, using softmax activation for classification.

### 10. **Model Compilation and Training**
   - Stochastic Gradient Descent (SGD) with Nesterov momentum is used for training the model, optimizing it for accuracy in classifying user input intents.

### 11. **Model Saving and Deployment**
   - Once the model is trained on the data, it is saved as `chatbot_model.h5` for future use in responding to user queries during real-time interaction.
