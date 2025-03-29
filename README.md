# Sentiment Analysis on Book Reviews

## Overview
This project performs **sentiment analysis** on Amazon Kindle book reviews using **Natural Language Processing (NLP)** and **Machine Learning**. The goal is to classify reviews as **positive, neutral, or negative** based on the text. The models implemented include **Random Forest Classifier, Gradient Boosting Classifier, Multinomial Naïve Bayes, and Support Vector Machine (SVM)**. Each model is evaluated based on **accuracy, precision, recall, and F1-score**, with **Random Forest** achieving the best performance and being saved for future predictions.

## Features
- **Data Preprocessing**: Cleans and processes text using **SpaCy** (lemmatization, stopword removal, punctuation removal).
- **TF-IDF Vectorization**: Converts text into numerical features.
- **Model Training & Evaluation**: Uses multiple machine learning models to classify sentiment.
- **Performance Metrics & Confusion Matrix**: Evaluates model effectiveness.
- **Model Saving**: Stores the best-performing model (`Random Forest`) for real-world use.

## Models Implemented
- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **Multinomial Naïve Bayes**
- **Support Vector Machine (SVM)**

## Tech Stack
- **Programming Language:** Python
- **Libraries:**
  - **Data Handling:** Pandas, NumPy
  - **NLP & Text Processing:** SpaCy, Gensim, Scikit-learn
  - **Machine Learning:** Random Forest, Gradient Boosting, Naïve Bayes, SVM
  - **Visualization:** Matplotlib, Seaborn
  - **Model Persistence:** Pickle (for saving models & vectorizers)

## Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis.git
   cd sentiment-analysis
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the script**
   ```bash
   python Sentiment_Analysis.py
   ```

## Results
The **Random Forest classifier** achieved the best performance in classifying sentiments. The script provides a detailed **comparison of all models** along with a **confusion matrix visualization**.

## Usage
- The trained model can be used to classify new book reviews.
- The code can be modified to analyze sentiments on other textual datasets.

## License
This project is open-source and available under the **MIT License**.


