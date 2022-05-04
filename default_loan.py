import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


"""
ML Model built to determine whether a customer will pay back their loan
97% Precision
"""

# Pandas dataframe display options for pycharm output
desired_width = 410
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', None)

df = pd.read_csv('lending_club_loan_two.csv')
# print(df.head())

"""
EXPLORATORY DATA ANALYSIS
"""

# We will be predicting "Fully Paid" loans vs. "Charged Off" loans
# sns.countplot(data=df, x='loan_status')

# "Fully Paid" = 1, "Charged Off" = 0
df['loan_status'] = df['loan_status'].replace(['Fully Paid', 'Charged Off'], [1, 0])
# Converting term length with numbers
# print(df['term'].unique())
df['term'] = df['term'].replace([' 36 months', ' 60 months'], [36, 60])

# Appears loan_status has a significant negative correlation with interest rate (-0.25) but nothing else of that level
plt.figure(figsize=(12, 7))
# sns.heatmap(df.corr(), annot=True, cmap='viridis')
# plt.show()

"""
DATA PRE-PROCESSING
"""

# Missing values in emp_title, emp_length, title, revol_util, mort_acc, & pub_rec_bankruptcies
df.isnull().sum()

# Numerical columns with missing values will be replaced with mean of values in column.
df['revol_util'].fillna(value=df['revol_util'].mean(), inplace=True)
df['mort_acc'].fillna(value=df['mort_acc'].mean(), inplace=True)
df['pub_rec_bankruptcies'].fillna(value=df['pub_rec_bankruptcies'].mean(), inplace=True)

# Dropping categorical values with too many unique values to turn into dummy variables
# Both emp_title & title have too many unique values to turn into dummy variables so these will be dropped
df = df.drop(columns=['emp_title', 'title', 'emp_title', 'issue_d', 'grade', 'emp_length'], axis=1)

# Converting categorical values into dummy values
dummies = pd.get_dummies(df[['verification_status', 'application_type', 'initial_list_status', 'purpose']],
                         drop_first=True)
df = df.drop(['verification_status', 'application_type', 'initial_list_status', 'purpose'], axis=1)
df = pd.concat([df, dummies], axis=1)

# Subgrade
subgrade_dummies = pd.get_dummies(df['sub_grade'], drop_first=True)
df = pd.concat([df.drop('sub_grade', axis=1), subgrade_dummies], axis=1)

# Home Ownership
# Replacing none, and any with other, so I only end up with four columns: MORTGAGE, RENT, OWN, OTHER
df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')
dummies = pd.get_dummies(df['home_ownership'], drop_first=True)
df = df.drop('home_ownership', axis=1)
df = pd.concat([df, dummies], axis=1)

# Using only zip codes from address
df['zip_code'] = df['address'].apply(lambda address: address[-5:])
dummies = pd.get_dummies(df['zip_code'], drop_first=True)
df = df.drop(['zip_code', 'address'], axis=1)
df = pd.concat([df, dummies], axis=1)

# Using only the year from credit line
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date: int(date[-4:]))
df = df.drop('earliest_cr_line', axis=1)

"""
MODEL CREATION
"""
X = df.drop('loan_status', axis=1).values
y = df['loan_status'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()

model.add(Dense(78,  activation='relu'))
model.add(Dropout(0.2))
# hidden layer
model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))
# hidden layer
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))
# output layer
model.add(Dense(units=1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=X_train,
          y=y_train,
          epochs=25,
          batch_size=256,
          validation_data=(X_test, y_test),
          )

predictions = model.predict_classes(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
