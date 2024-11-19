# PREDICTING-MEDICAL-CONDITIONS-FROM-REDDIT-POSTS
Modern day social media is filling up with discussions revolving around one’s or a close one’s health and such a platform can be of significant help when it comes to pre-diagnosis.
This study applies machine learning approach to make a discovery about medical situation by analyzing text data from Reddits.
Thus, the exploratory analysis of four ML models: Logistic Regression, LSTM, Decision Tree, and Random Forest was performed with the help of feature extraction using TF-IDF and Word2Vec. Logistic Regression had the highest accuracy score of 68.7/100; LSTM since it is a model that works best for sequential input data had an accuracy of 71.8/100. The analysis shows that more basic algorithms, such as Logistic Regression, and sequence-sensitive models like LSTM, are feasible for health data categorization from social media.
Project Title: Predicting Medical Conditions from Reddit Posts
Project Description: Develop a machine learning model to predict specific medical conditions from text data sourced from Reddit posts. The goal is to create a tool that can assist in early detection and classification of various health conditions by analyzing patterns and content in social media posts.
Dataset: Reddit Medical Conditions Dataset(https://zenodo.org/records/3941387#.YFfi3EhJHL8)
Description: This dataset contains Reddit posts related to various medical conditions, labeled with specific conditions or symptoms. It is suitable for training models to classify medical conditions based on text data.
Steps to be Completed in this Project:
Data Preprocessing:
Load Data: Import Reddit posts from the dataset, including post text, metadata (e.g., post time, user information), and labels indicating mental health conditions.
Text Cleaning: Perform text preprocessing, including removing special characters, URLs, and stop words. Tokenize and normalize the text.
Feature Extraction: Convert text data into numerical features using techniques like TF-IDF, word embeddings (e.g., Word2Vec, GloVe), or language models (e.g., BERT).
Mental Health Crisis Detection:
Model Development:
Train a classification model (e.g., Logistic Regression, Random Forest, or a deep learning model like LSTM or BERT) to detect posts indicative of mental health crises.
Experiment with different architectures and hyperparameters to optimize performance.
Training:
Split the dataset into training, validation, and test sets. Train the model on the training set and tune parameters based on the validation set.
Model Evaluation:
Metrics:
Evaluate the model using metrics such as Accuracy, Precision, Recall, F1-score, and ROC-AUC to assess its performance in identifying mental health crises.
Error Analysis:
Perform error analysis to understand model limitations and refine the model based on common misclassifications.
