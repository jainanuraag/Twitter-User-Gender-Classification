# B351-Group-7-Final-Project
The repository for B351 Intro to AI final project.

This project uses various Machine Learning models to predict Twitter user gender from a given tweet. The following dataset from Kaggle was used and is available for download: https://www.kaggle.com/crowdflower/twitter-user-gender-classification.

The linguistic composition of tweets was used to train and use machine learning models to classify tweets by gender. To view predictions, a user can follow the steps below:
  1. Clone or download the repository (cloning link: https://github.iu.edu/wangbote/B351-Group-7-Final-Project.git)
  2. Open a command line tool on the user's computer and navigate to the cloned/downloaded repository
  3. Use the following command: python GUI.py (if Python is not installed, it can be downloaded from https://www.python.org/downloads/)
  4. When the user interface opens, type the text of a tweet into the top left of the interface
  5. The 'Submit Tweet' button shows gender classification based on this project's implementation of the Naive Bayes classifier
  6. The 'Show Sklearn Prediction' button shows gender classification using a 3rd party model
  7. The 'Validation Data Accuracy' button shows the accuracy of this project's implementation of the Naive Bayes classifier on validation data
  
To view a comparison of models, the below steps must be followed:
  1. Open a command line tool on the user's computer and navigate to the cloned/downloaded repository
  2. Use the following command: python ModelComparison.py
  3. The accuracy scores of various models are shown for the user to compare models
