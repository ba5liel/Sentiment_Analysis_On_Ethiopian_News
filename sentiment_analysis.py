import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing.sequence import pad_sequences

class SentimentModel:
    def __init__(self, max_features=10000, maxlen=500):
        """
        Initialize the SentimentModel class for sentiment analysis.

        Args:
            max_features (int): Number of words to consider as features.
            maxlen (int): Maximum length of the sequences.
        """
        self.max_features = max_features
        self.maxlen = maxlen
        self.model = self.build_model()
        self.word_index = self.load_word_index()

    def build_model(self):
        """
        Build the sentiment analysis model.

        Returns:
            A compiled Keras model.
        """
        model = Sequential()
        model.add(Embedding(self.max_features, 128))
        model.add(LSTM(32))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def load_data(self):
        """
        Load and preprocess the IMDb dataset.

        Returns:
            Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
        """
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=self.max_features)
        x_train = pad_sequences(x_train, maxlen=self.maxlen)
        x_test = pad_sequences(x_test, maxlen=self.maxlen)
        return (x_train, y_train), (x_test, y_test)
    def load_word_index(self):
        """
        Load the IMDb dataset's word index and adjust it for use with the embedding layer.

        Returns:
            dict: A dictionary mapping words to their integer indices.
        """
        # Load the original word index from IMDb
        original_word_index = imdb.get_word_index()
        
        # Adjust the word index, shifting it by 3 because Keras reserves indices 0, 1, and 2
        # 0 is usually reserved for padding, 1 for the start of the sequence, and 2 for unknown words (out-of-vocabulary)
        adjusted_word_index = {word: index + 3 for word, index in original_word_index.items()}
        
        # Add special tokens to the adjusted word index
        adjusted_word_index["_PAD"] = 0 # Index for padding
        adjusted_word_index["_START"] = 1 # Index for the start of a sequence
        adjusted_word_index["_UNK"] = 2 # Index for unknown words (out-of-vocabulary)
        return adjusted_word_index

    def train(self, x_train, y_train, x_val, y_val, epochs=10, batch_size=64):
        """
        Train the model.

        Args:
            x_train, y_train: Training data and labels.
            x_val, y_val: Validation data and labels.
            epochs (int): Number of epochs to train.
            batch_size (int): Batch size for training.

        Returns:
            A history object containing the training history.
        """
        return self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                              validation_data=(x_val, y_val))

    def evaluate(self, x_test, y_test):
        """
        Evaluate the model on the test data.

        Args:
            x_test (np.ndarray): Test data.
            y_test (np.ndarray): Test labels.

        Returns:
            list: Evaluation results.
        """
        return self.model.evaluate(x_test, y_test)

    def predict(self, text):
        """
        Predict the sentiment of a given text.

        Args:
            text (str): Text to predict sentiment for.

        Returns:
            float: The predicted sentiment score.
        """
        # Preprocess and predict logic here
        pass

    def save(self, model_dir='sentiment_model.h5'):
        """
        Save the model to a file.

        Args:
            model_dir (str): Path to save the model.
        """
        self.model.save(model_dir)

    def load(self, model_dir='sentiment_model.h5'):
        """
        Load the model from a file.

        Args:
            model_dir (str): Path to load the model from.
        """
        self.model = tf.keras.models.load_model(model_dir)

class SentimentPredictor:
    def __init__(self, sentiment_model: SentimentModel):
        """
        Initialize the SentimentPredictor class.

        Args:
            sentiment_model (SentimentModel): An instance of the SentimentModel class.
        """
        self.sentiment_model = sentiment_model

    def preprocess_text(self, text):
        """
        Convert input text to a sequence of integers.

        Args:
            text (str): Text input.

        Returns:
            np.ndarray: Processed text data ready for model input.
        """
        oov_index =2
        max_features = self.sentiment_model.max_features
        tokens = text.lower().split()
        word_index = self.sentiment_model.word_index
        sequence = [word_index.get(word, oov_index) if word_index.get(word, max_features) < max_features else oov_index for word in tokens]  # 2 is typically the OOV index
        padded_sequence = pad_sequences([sequence], maxlen=self.sentiment_model.maxlen)
        return padded_sequence

    def predict(self, text):
        """
        Predict the sentiment of a given text.

        Args:
            text (str): Text to predict sentiment for.

        Returns:
            float: The predicted sentiment score, where scores closer to 1 indicate positive sentiment.
        """
        # Preprocess text to fit model input
        processed_text = self.preprocess_text(text)
        
        # Perform prediction
        prediction = self.sentiment_model.model.predict(processed_text)
        
        # Return the prediction result
        return prediction[0][0]