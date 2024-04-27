# Sentiment-Analysis-On-Ethiopian-News

This project aims to train a machine learning model to categorize news articles, leveraging TensorFlow, Keras, and the Google Translate API. The model scrapes, translates, and classifies news articles into categories based on their sentiment.

## Project Structure
- **`sentiment_classification.py`**: Defines the `SentimentModel`, `SentimentTrainer`, and `SentimentPredictor` classes for building, training, and using the sentiment classification model.
- **`news_scraper_translator.py`**: Includes `NewsScraper` and `TextTranslator` classes for extracting and translating article content.
- **`demo.py`**: Script for training the model and demonstrating its use on news articles.

## Usage
To use this project, install the necessary libraries with `pip install -r requirements.txt`, then run `demo.py` to train the model and classify a sample news article.

## Limitations and Future Enhancements
### Current Limitations
- **Domain-Specific Training**: The model may not perform well outside the specific news domains it was trained on, due to differences in language usage and topics.
- **Language Limitations**: Currently, the translation and classification are optimized for English. Performance may degrade with different languages or nuanced linguistic features.
- **Scalability**: The model's performance and speed might decrease as the size and frequency of the data increase.

### Future Enhancements
1. **Advanced Model Techniques**: Implement state-of-the-art models like BERT or RoBERTa for better context understanding and sentiment analysis.
2. **Multi-Language Support**: Enhance the `TextTranslator` class to handle multiple languages more robustly, possibly using advanced detection and translation models.
3. **Dynamic Scraping Capabilities**: Improve the `NewsScraper` class to adapt to various website layouts and content formats dynamically.
4. **Real-Time Classification**: Develop a system to monitor and classify news articles in real-time as they are published.
5. **User Interface**: Create a web interface or API that allows users to input URLs and receive classifications directly.
6. **Model Optimization**: Introduce model caching and more efficient data handling techniques to reduce load times and computational demands.
7. **Comprehensive Evaluation**: Expand evaluation metrics to include precision, recall, and F1 scores to provide a more detailed assessment of model performance.