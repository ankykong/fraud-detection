# fraud-detection
credit card fraud detection

## Abstract
The goal was to create a model that would be able to determine if a card transaction was fraudulent or not. The model was developed using thousands of card transactions with 28 unidentified variables, the amount, and transaction time. I then took in information from a user. I converted
the information into a similar structure to determine if the new transaction input by the user would be fraudulent or not.

## Data Description
The dataset consists of several attributes (V1-V28) containing coded information about the transaction to help determine if a transaction was a valid or fraudulent charge. There is also the time and amount that was transacted and lastly, each row is labeled if it was fraudulent or not.

## Algorithm
This project uses logistic regression to determine if the transactions should be coded as fraudulent or legitimate. It essentially draws a line between the dataset in a 30D space and everything on one side is considered to be fraudulent and everything on the other is legit.

## Tools Used
I used **scikitlearn** to implement the logistic regression model. I used **pandas** to manipulate and interact with the dataset as an easy-to-use data frame. I used an **undersampler** to get better results. I used a GridSearchCV as a model selector to determine the best logistic regression model. I used **train_test_split** to split the data into training and testing data. I used **Streamlit** to host the model online along with a form to check transactions.

Here's the streamlit application: https://fraud-detection-zfkmhdbd3xnjymj7wptpue.streamlit.app/
