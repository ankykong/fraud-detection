import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context='notebook', style='whitegrid', font_scale=1.2)
import streamlit as st
import os
import boto3
from dotenv import load_dotenv
from botocore.config import Config


# ------------------------------------------------------
#                      APP CONSTANTS
# ------------------------------------------------------
REMOTE_DATA = 'creditcard.csv'


# ------------------------------------------------------
#                        CONFIG
# ------------------------------------------------------
load_dotenv()

# Load Backblaze connection credentials from environment variables
b2_endpoint = os.environ['B2_ENDPOINT']
b2_key_id = os.environ['B2_KEYID']
b2_secret_key = os.environ['B2_APPKEY']
b2_bucketname = os.environ['B2_BUCKETNAME']

# Function to configure Boto3 to use with Backblaze B2
def configure_boto3(key_id, application_key):
    return boto3.Session(
        aws_access_key_id=key_id,
        aws_secret_access_key=application_key
    )

# Function to load data from Backblaze B2
def load_data_from_b2(session, bucket_name, object_name):
    s3 = session.resource('s3', endpoint_url=b2_endpoint)
    obj = s3.Object(bucket_name, object_name)
    return pd.read_csv(obj.get()['Body'])

# Configure Boto3 with Backblaze B2 credentials
session = configure_boto3(b2_key_id, b2_secret_key)
df = load_data_from_b2(session, b2_bucketname, REMOTE_DATA) 
df = df.dropna()

st.write(
'''
## Credit Card Fraudulent/Non Fraudulent Transactions
We pull data from our Backblaze storage bucket, and render it in Streamlit using `st.dataframe()`.
''')

st.dataframe(df)

fraud = df[df['Class']==1]
non_fraud = df[df['Class']==0]
st.dataframe(fraud['Amount'])
st.dataframe(non_fraud)

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(14,6))
ax1.set_title('Fraud')
ax1.set_ylabel("Amount ($)")
box_plot = sns.boxplot(data=fraud['Amount'], ax=ax1)

median = fraud['Amount'].median()
vertical_offset = median * 0.05
for xtick in box_plot.get_xticks():
    box_plot.text(xtick, median + vertical_offset,median, 
            horizontalalignment='center',size='x-small',color='red',weight='semibold')

ax2.set_title('Non-Fraud')
ax2.set_ylabel("Amount ($)")
box_plot = sns.boxplot(data=non_fraud['Amount'], ax=ax2)

show_graph = st.checkbox('Show Graph', value=True)
if show_graph:
    st.pyplot(fig)

median = non_fraud['Amount'].median()
vertical_offset = median * 0.05
for xtick in box_plot.get_xticks():
    box_plot.text(xtick, median + vertical_offset,median, 
            horizontalalignment='center',size='x-small',color='red',weight='semibold')
    
medians = [fraud['Amount'].median(), non_fraud['Amount'].median()]
means = [fraud['Amount'].mean(), non_fraud['Amount'].mean()]
ranges = [(min(fraud['Amount']),max(fraud['Amount'])), (min(non_fraud['Amount']), max(non_fraud['Amount']))]
st.write(f'Fraud Median: {medians[0]} \t \t Non-Fraud Median: {medians[1]}')
st.write(f'Fraud Mean: {means[0]} \t Non-Fraud Mean: {means[1]}')
st.write(f'Fraud Range: {ranges[0]} \t Non-Fraud Range: {ranges[1]}')
