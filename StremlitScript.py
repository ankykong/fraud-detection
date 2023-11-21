import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context='notebook', style='whitegrid', font_scale=1.2)
import streamlit as st
from utils.b2 import B2


# ------------------------------------------------------
#                      APP CONSTANTS
# ------------------------------------------------------
REMOTE_DATA = 'seattle_home_prices.csv'


# ------------------------------------------------------
#                        CONFIG
# ------------------------------------------------------
load_dotenv()

# load Backblaze connection
b2 = B2(endpoint=os.environ['B2_ENDPOINT'],
        key_id=os.environ['B2_KEYID'],
        secret_key=os.environ['B2_APPKEY'])

st.write(
'''
## Credit Card Fraudulent/Non Fraudulent Transactions
We pull data from our Backblaze storage bucket, and render it in Streamlit using `st.dataframe()`.
''')

df = pd.read_csv("./creditcard.csv")
st.dataframe(df)

fraud = df[df['Class']==1]
non_fraud = df[df['Class']==0]

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