import os
import re

from news_scraper_translator import NewsScraper, TextTranslator
from sentiment_analysis import SentimentModel, SentimentPredictor

class ModelTrainer:
    def __init__(self, sentiment_model: SentimentModel):
        self.sentiment_model = sentiment_model

    def train_model(self):
        (train_data, train_labels), (test_data, test_labels) = self.sentiment_model.load_data()
        # Split some data for validation
        x_val = train_data[:10000]
        y_val = train_labels[:10000]
        partial_x_train = train_data[10000:]
        partial_y_train = train_labels[10000:]
        # Train the model
        history = self.sentiment_model.train(partial_x_train, partial_y_train, x_val, y_val)
        # Evaluate the model on the test data
        results = self.sentiment_model.evaluate(test_data, test_labels)
        return history, results

class NewsAnalyzer:
    def __init__(self, url: str, sentiment_model: SentimentModel) -> None:
        """
        Initialises the NewsAnalyzer class.

        Args:
            url (str): URL of the news article to be analysed.
            sentiment_model (SentimentModel): An instance of the SentimentModel class.

        """
        self.url = url
        self.sentiment_model = sentiment_model
        self.news_scraper = NewsScraper(self.url)
        self.translator = TextTranslator()
        self.predictor = SentimentPredictor(self.sentiment_model)

    def analyze_news(self) -> str:
        """
        Analyses the news article and predicts its label.

        Returns:
            str: Predicted label for the news article.

        """
        # Fetch news data using scraper
        self.news_scraper.fetch_data()
        title = self.news_scraper.extract_title()
        content = self.news_scraper.extract_content()
        content = """An incredible movie. One that lives with you.
It is no wonder that the film has such a high rating, it is quite literally breathtaking."""
        # Translate content to desired language (assuming English)
        content = title
        content = content[:1000] if len(content) > 1000 else content
        translated_text = self.translator.translate(content)
        translated_text = self.clean_text(translated_text)
        print(f"the input: {translated_text}")
        
        # Predict the label for the translated news content
        predicted_label = self.predictor.predict(translated_text)
        return predicted_label
    
    def clean_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove URLs, handles, and the hashtag symbol
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#','', text)
        # Replace punctuation with space
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        return text
if __name__ == "__main__":
    model_path = 'sentiment_model.h5'
    
    # Check if model is saved in file and if it's load it; if not, start training
    if os.path.exists(model_path):
        sentiment_model = SentimentModel()
        sentiment_model.load(model_path)
        print("Model loaded successfully.")
    else:
        sentiment_model = SentimentModel()
        model_trainer = ModelTrainer(sentiment_model)
        history, results = model_trainer.train_model()
        print("Training completed with accuracy:", results[1])
        # Save model here
        sentiment_model.save(model_path)
        print("Model saved to", model_path)

    # Analyse a news article and predict its label
    news_analyzer = NewsAnalyzer("https://www.ena.et/web/amh/w/amh_4350218", sentiment_model)
    predicted_label = news_analyzer.analyze_news()
    print("Predicted label:", predicted_label)
