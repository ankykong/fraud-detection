import os
import json
import pickle
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from utils.b2 import B2


# ------------------------------------------------------
#                      APP CONSTANTS
# ------------------------------------------------------
REMOTE_DATA = 'creditcard.csv'


# ------------------------------------------------------
#                        CONFIG
# ------------------------------------------------------
load_dotenv()

# load Backblaze connection
b2 = B2(endpoint=os.environ['B2_ENDPOINT'],
        key_id=os.environ['B2_KEYID'],
        secret_key=os.environ['B2_APPKEY'])


# ------------------------------------------------------
#                         APP
# ------------------------------------------------------
st.write(
'''
## Credit Card Fraudulent/Non Fraudulent Transactions
We pull data from our Backblaze storage bucket, and render it in Streamlit using `st.dataframe()`.
''')

b2.set_bucket(os.environ['B2_BUCKETNAME'])

df_card_transactions = b2.to_df(REMOTE_DATA)
st.dataframe(df_card_transactions)