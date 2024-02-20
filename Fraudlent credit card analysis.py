#!/usr/bin/env python
# coding: utf-8

# In[24]:


get_ipython().system('pip install imblearn')


# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


df = pd.read_csv('creditcard.csv')


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.duplicated().any()


# In[10]:


# Drop the duplicate rows from the DataFrame
df = df.drop_duplicates()
#Reset index
df = df.reset_index(drop=True)


# In[11]:


df.duplicated().any()


# In[12]:


df.Class.value_counts()


# In[13]:


df2=df.copy()


# EDA

# In[14]:


plt.figure(figsize=(10,5))
plt.plot(df2.index,df2["Time"])
plt.xlabel("Transaction")
plt.ylabel("Time")
plt.title("Distribution of Transaction Time")
plt.show()


# In[15]:


plt.figure(figsize=(10,5))
plt.plot(df2.index,df2["Amount"])
plt.xlabel("Transaction")
plt.ylabel("Amount")
plt.title("Distribution of Transaction Amount")
plt.show()


# In[16]:


corr = df2.corr()
c=corr['Class'].sort_values(ascending=False)
print(c)


# In[17]:


# value of 1 in the "Class" column indicates (fraud) transactions
fraud=df2[df2["Class"]==1]
# value of 0 in the "Class" column indicates (real) transactions
real=df2[df2["Class"]==0]


# In[18]:


plt.figure(figsize=(10,5))
plt.scatter(real["Time"],real["V11"],label="Real Transaction")
plt.scatter(fraud["Time"],fraud["V11"],label="Fraud Transaction")
plt.xlabel("Time")
plt.ylabel("V11")
plt.title("Max positive correlation(V11)")
plt.legend()
plt.show()


# In[19]:


plt.figure(figsize=(10,5))
plt.scatter(real["Time"],real["V17"],label="Real Transaction")
plt.scatter(fraud["Time"],fraud["V17"],label="Fraud Transaction")
plt.xlabel("Time")
plt.ylabel("V17")
plt.title("Max negative correlation(V17)")
plt.legend()
plt.show()


# In[20]:


from sklearn.preprocessing import StandardScaler
# standardizes the 'Amount' column in the DataFrame 
df2['amount'] = StandardScaler().fit_transform(df2['Amount'].values.reshape(-1,1))
# standardizes the 'Tine' column in the DataFrame 
df2['time'] = StandardScaler().fit_transform(df2['Time'].values.reshape(-1,1))
# Removng this non-scaling columns
df2.drop(['Time','Amount'], axis=1, inplace=True)
df2


# Feature selection

# In[21]:


X = df2.drop('Class', axis=1)
Y = df2['Class']


# In[25]:


from imblearn.under_sampling import RandomUnderSampler
undersampler = RandomUnderSampler(random_state=42)
X_undersampled, y_undersampled = undersampler.fit_resample(X, Y)


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_undersampled,y_undersampled, test_size = 0.25, random_state=42)


# Logistic Regression

# In[27]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)


# In[28]:


from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score, classification_report
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model with  accuracy, F1 score, precision and recall
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
classifcation_report_logistic = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
print("Classification Report:")
print(classifcation_report_logistic)


# In[29]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16}, cbar=False, square=True)

# Add labels and titles
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

# Show the plot
plt.show()


# SVM
# 

# In[30]:


from sklearn.svm import SVC
# Initialize SVM Classifier
svm_classifier = SVC( kernel='rbf', random_state=42)
# Train the SVM classifier on the training data
svm_classifier.fit(X_train,y_train)
# Make predictions on the testing data
y_pred_svm = svm_classifier.predict(X_test)
# Evaluate the model with  accuracy, F1 score, precision and recall
accuracy = accuracy_score(y_test,y_pred_svm)
f1 = f1_score(y_test, y_pred_svm, average='weighted')
precision = precision_score(y_test,y_pred_svm, average='weighted')
recall = recall_score(y_test, y_pred_svm, average='weighted')
classifcation_report_ = classification_report(y_test, y_pred_svm)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
print("Classification Report:")
print(classifcation_report_)
# confusion matrix for model performance assessment
cm = confusion_matrix(y_test,y_pred_svm)
# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16}, cbar=False, square=True)
# Add labels and titles
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
# Show the plot
plt.show()


# Random Forest

# In[31]:


from sklearn.ensemble import RandomForestClassifier
# Initialize RandomFores Classifier
rf = RandomForestClassifier(random_state=42)
# train the model
rf.fit(X_train,y_train)
# Make predictions on the test set
y_pred_rf = rf.predict(X_test)
# Evaluate the model with  accuracy, F1 score, precision and recall
accuracy = accuracy_score(y_test,y_pred_rf)
f1 = f1_score(y_test, y_pred_rf, average='weighted')
precision = precision_score(y_test,y_pred_rf, average='weighted')
recall = recall_score(y_test, y_pred_rf, average='weighted')
classifcation_report_ = classification_report(y_test, y_pred_rf)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
print("Classification Report:")
print(classifcation_report_)
# confusion matrix for model performance assessment
cm = confusion_matrix(y_test,y_pred_rf)
# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16}, cbar=False, square=True)
# Add labels and titles
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
# Show the plot
plt.show()


# In[ ]:




