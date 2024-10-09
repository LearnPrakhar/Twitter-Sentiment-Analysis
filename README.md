# Project Overview
This project is a sentiment analysis of a Twitter dataset, where we aim to classify tweets into categories such as "Positive" or "Negative" sentiments. The steps include data preprocessing, text cleaning, feature extraction using TF-IDF, and model training with different machine learning algorithms.

# Workflow
The notebook follows the general steps outlined below:

Data Loading:

Load the Twitter dataset (twitter_training.csv).
Data Preprocessing:

Remove unnecessary columns (ID, Platform).
Handle missing values by removing rows with NaN values.
Remove duplicate entries to ensure data quality.
Text Cleaning:

Clean the tweet text by removing URLs, mentions, hashtags, special characters, and converting the text to lowercase.
This is done using regular expressions.
Feature Extraction (TF-IDF):

Transform the cleaned tweet text into numerical features using TfidfVectorizer, which converts the text into a matrix of TF-IDF features.
The vectorizer is limited to the top 5000 features and removes English stop words.
Model Training:

Multiple machine learning models are used to classify the sentiment:
Logistic Regression
Naive Bayes
Support Vector Machines (SVM)
Hyperparameter tuning is performed using GridSearchCV for Naive Bayes and SVM to improve the accuracy of the models.
Model Evaluation:

The models are evaluated based on accuracy, precision, recall, and F1-score using a test dataset.
File Structure
Sentiment Analysis.ipynb: Jupyter notebook containing the code for sentiment analysis.
# Requirements
To run this notebook, you'll need the following libraries installed:

pip install numpy pandas matplotlib seaborn scikit-learn nltk
Optional for visualizations:

pip install plotly wordcloud
How to Run the Project
Clone the repository (or download the notebook):

Place the dataset twitter_training.csv in the working directory.
Install the dependencies: Run the following command to install the required libraries:

pip install -r requirements.txt
Run the Jupyter Notebook: Open the notebook and execute the cells step-by-step to perform sentiment analysis on the dataset. You can launch Jupyter by running:

jupyter notebook Sentiment Analysis.ipynb
Results
After running the notebook, you'll get:

leaned and preprocessed text data ready for analysis.
A comparison of the performance of various models (Logistic Regression, Naive Bayes, and SVM).
