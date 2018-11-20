
# coding: utf-8

# In[1]:


'''Downloading stopwords'''
import nltk
nltk.download('stopwords')


# In[5]:


'''Function to tokenize, remove stop words and create a bigram model'''
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
def process_message(message):
    gram=2
    stop_words = set(stopwords.words('english')) 
    try:
        word_tokens = word_tokenize(message) 
    except:
        return
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    filtered_sentence = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    words=[w for w in filtered_sentence if len(w)>2]
    w=[]
    for i in range(len(words)-gram+1):
        w+=[''.join(words[i:i+gram])]
    return w


# In[6]:


'''Read the dataset'''
import pandas as pd
df=pd.read_csv('spam.csv')
df.v2 = df.v2.astype(str).str.lower()
X=df.iloc[:, 1].values
y=df.iloc[:,0].values
from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
y=label_encoder.fit_transform(y)
y.size


# In[7]:


'''Processing the dataset'''
from collections import Counter
import numpy as np
numcol=X.size
X_processed=[]
bigram_set=set()
y_list=list()
for i in range (numcol):
    pm=process_message(X[i])
    if pm:
        y_list.append(y[i])
        for j in range(len(pm)):
            bigram_set.add(pm[j]) 
        X_processed.append(Counter(pm))
y_new=np.asarray(y_list)
y_new.size


# In[8]:


'''Creating bag of words of the bigram model'''
rows=len(X_processed)
cols=len(bigram_set)
matrix=np.zeros(shape=(rows,cols))
for i in range (rows):
    count=0
    for j in bigram_set:
        if j in X_processed[i]:
            #print(X_processed[i][j] )
            matrix[i][count]=X_processed[i][j]
        #print(matrix[i][count])    
        count+=1
matrix      


# In[9]:


'''Splitting dataset into training and testing data'''
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(matrix, y_new, test_size=0.3)


# In[10]:


'''Compressing the sparse matrix'''
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
sparse_dataset = csr_matrix(X_train)
sparse_dataset


# In[11]:


'''SVM Classifier'''
from sklearn.svm import SVC
model=SVC(gamma=2, C=1)
model.fit(sparse_dataset, y_train)
pr=model.predict(X_test)
score = accuracy_score(y_test, pr)
score


# In[14]:


'''Decision Tree Classifier'''
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
#model.fit(sparse_dataset, y_train)
#pr=model.predict(X_test)
#score = accuracy_score(y_test, pr)
from sklearn.model_selection import cross_val_score
print(cross_val_score(model, sparse_dataset, y_train, cv = 10, scoring = "accuracy"))


# In[15]:


'''Random Forest Classifier'''
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(sparse_dataset, y_train)
pr=model.predict(X_test)
score = accuracy_score(y_test, pr)
score
cross_val_score(model, sparse_dataset, y_train, cv = 10, scoring = "accuracy")


# In[38]:


'''Naive Bayes Classifier'''
from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(sparse_dataset, y_train)
y_predict=naive_bayes.predict(X_test)
score = accuracy_score(y_test, y_predict)
score


# In[39]:


'''Gradient Boost Classifier'''
from sklearn.ensemble import GradientBoostingClassifier
model= GradientBoostingClassifier(learning_rate=0.03,random_state=1)
model.fit(sparse_dataset, y_train)
pr=model.predict(X_test)
score = accuracy_score(y_test, pr)
score

