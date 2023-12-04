import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV

# Set plot style
plt.style.use('ggplot')

# Turn off warnings
import warnings
warnings.filterwarnings('ignore')
sns.set_theme(context='notebook', style='whitegrid', font_scale=1.2)
import streamlit as st


df = pd.read_csv('creditcard.csv')
df = df.dropna()

st.write(
'''
## Credit Card Fraudulent/Non Fraudulent Transactions
''')

data = df.drop(columns=['Class'])

answer = df['Class']

X_trainval, X_test, y_trainval, y_test = train_test_split(data, answer
                                                          , test_size=0.2
                                                          , stratify=df['Class']
                                                          , random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval
                                                  , test_size=0.25
                                                  , stratify=y_trainval
                                                  , random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_std = scaler.transform(X_train)
X_val_std = scaler.transform(X_val)
X_test_std = scaler.transform(X_test)

rus = RandomUnderSampler(random_state=42)

X_train_under, y_train_under = rus.fit_resample(X_train_std, y_train)

X_val_under, y_val_under = rus.fit_resample(X_val_std, y_val)

penalty = ['l2']
C = np.logspace(0, 4, 10, 100, 1000)
param_grid = dict(C=C, penalty=penalty)

logistic = linear_model.LogisticRegression(solver='lbfgs', max_iter=10000)
logistic_grid = GridSearchCV(logistic, param_grid, cv=5, scoring='roc_auc', verbose=10, n_jobs=-1)
logistic_grid.fit(X_train_under, y_train_under)

st.title("Check if Charge is Fraudulent")

with st.form(key="charge_from"):
    date = st.date_input("Date:")
    time = st.time_input("Time:")
    location = st.text_input("Location:")
    category = st.text_input("Category:")
    amount = st.number_input("Amount:")

    submitted = st.form_submit_button("Submit")

def create_variables_from_seed(seed_phrase):
        seed_value = hash(seed_phrase)

        ranges = [(min(df['V'+str(i)]), max(df['V'+str(i)])) for i in range(1, 29)]

        variables = [
            min(max(seed_value + i, rng[0]), rng[1]) for i, rng in enumerate(ranges)
        ]

        return variables

if submitted:
    
    phrase = location + " " + category
    created_variables = create_variables_from_seed(phrase)
    created_variables.insert(0, time.hour*360+time.minute*60+time.second)
    created_variables.append(amount)

    if logistic_grid.predict([created_variables]) == 1:
        st.write("Fraudulent")
    else:
        st.write("Not Fraudulent")
