#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd 
from pandas import DataFrame


# In[32]:


# converting the list of objects 
## take care of multiple classes for the same sentence
from re import search
import re
def filterTags(object):
    newObject = []
    # extract the sentence
    # extract the class tag
    if len(object) > 1:
        if object[1].strip("<").lower() == "class":
            newObject.append(object[0])
            # extract the id tag
            # extract the name tag
            for tagLine in object[2:]:
                if search("id", tagLine.split("=")[0]):
                    idWord = tagLine.split("=")[1]
                    idWord = re.sub(r'[^\w\s]', '', idWord) 
                    newObject.append(("id", idWord))
                if search("name", tagLine.split("=")[0]):
                    nameWords = tagLine.split("=")[1]
                    newObject.append(("name", nameWords))
                if search(">", tagLine):
                    break
            return(newObject)


# In[33]:


# labelling now
def creatingLabels(element):
    if element:
        sentence = element[0]
        for searchToken in element[1:]:
            if searchToken[0] == "id":
                idToken = searchToken[1]
                sentenceSplit = sentence.split(idToken)
                wordSplit1 = sentenceSplit[0].rstrip().split(" ")
                s1 = ""

                for w in wordSplit1:
                     s1 += w + "/O "
                sentenceSplit[0] = s1.rstrip()

                idTokenLabel = " " + idToken + "/B "

                if len(sentenceSplit) >= 2:
                    wordSplit2 = sentenceSplit[1].lstrip().split(" ")
                    s2 = ""
                    for w in wordSplit2:
                        s2 += w + "/O "
                    sentenceSplit[1] = s2.rstrip()
                    sentence = sentenceSplit[0] + idTokenLabel + sentenceSplit[1]
            
            if searchToken[0] == "name":
                nameToken = searchToken[1]
                sentenceSplit = sentence.split(nameToken)
                wordSplit1 = sentenceSplit[0].rstrip().split(" ")
                s1 = ""
                for w in wordSplit1:
                     s1 += w + "/O "
                sentenceSplit[0] = s1.rstrip()
                nameTokenList = nameToken.split(" ")
                nameTokenLabelOne = nameTokenList[0] + "/B "
                nameTokenLabelTwo = ''
                for t in nameTokenList[1:]:
                    nameTokenLabelTwo += t + "/I "  
                nameTokenLabelTwo.rstrip()  
                if len(sentenceSplit) >= 2: 
                    wordSplit2 = sentenceSplit[1].lstrip().split(" ")
                    s2 = ""
                    for w in wordSplit2:
                        s2 += w + "/O "
                    sentenceSplit[1] = s2.rstrip()
                #     sentence = sentenceSplit[0] + nameTokenLabelOne + nameTokenLabelTwo #+ sentenceSplit[1] 
                # else:
                    sentence = sentenceSplit[0] + nameTokenLabelOne + nameTokenLabelTwo + sentenceSplit[1] 
                
        return(sentence)


# In[34]:


def creatingWords(w):
    temp = []
    if w:
        for i in w.split(" "):
            temp.append(i)
        return temp
    else: return None

def extractingLables(w):
    temp = []
    if w:
        for i in w.split(" "):
            if len(i.split("/")) < 2:
                print(i)
            else:
                temp.append(i.split("/")[1])
        return temp
    else: return None
# creatingWords(labelled_list[0])


# In[35]:


preText = ''
with open('NLU.train', 'r') as file:
    preText = file.read()
inputData = preText.strip().split("\n\n")
df = pd.DataFrame(inputData)


# In[36]:


inputDataDF = df.apply (lambda x: x[0].strip().split("\n"), axis = 1)
inputDataDF = inputDataDF.apply (lambda x: filterTags(x))
input_list = inputDataDF.to_list()


# In[37]:


labelled_list = list(map(lambda x: creatingLabels(x), input_list))
filtered_labelled_list = [i for i in labelled_list if i] 
labelledDFTrain = DataFrame (filtered_labelled_list, columns = ["sentence"])
# labelled_list


# In[38]:


tokens = []
for l in labelled_list[:]:
    if creatingWords(l):
        tokens.extend(creatingWords(l))

wordDFTrain = DataFrame(tokens, columns = ["words"])
wordDFTrain = wordDFTrain[wordDFTrain.columns[0]].str.split("/", n = 1, expand = True)
wordDFTrain.columns = ['tokens', "labels"]


# In[39]:


# testing
preText = ''
with open('NLU.test', 'r') as file:
    preText = file.read()
inputDataTest = preText.strip().split("\n\n")
dfTest = pd.DataFrame(inputDataTest)


# In[40]:


inputDataDFTest = df.apply (lambda x: x[0].strip().split("\n"), axis = 1)
inputDataDFTest = inputDataDFTest.apply (lambda x: filterTags(x))
input_list_test = inputDataDFTest.to_list()


# In[41]:


labelled_list_test = list(map(lambda x: creatingLabels(x), input_list_test))
filtered_labelled_list_test = [i for i in labelled_list_test if i] 
labelledDFTest = DataFrame (filtered_labelled_list_test, columns = ["sentence"])


# In[42]:


tokens = []
for l in labelled_list[:]:
    if creatingWords(l):
        tokens.extend(creatingWords(l))

wordDFTest = DataFrame(tokens, columns = ["words"])
wordDFTest = wordDFTest[wordDFTest.columns[0]].str.split("/", n = 1, expand = True)
wordDFTest.columns = ['tokens', "labels"]


# In[43]:


# creating Feature 1
wordDFTrain["value"] = wordDFTrain.apply(lambda x: re.sub(r'[^\w\s]', '', x["tokens"]) , axis = 1)
wordDFTest["value"] = wordDFTest.apply(lambda x: re.sub(r'[^\w\s]', '', x["tokens"]) , axis = 1)


# In[44]:


# creating feature 2
wordDFTrain["AllUpperCase"] = wordDFTrain.apply(lambda x: x["value"].isupper(), axis = 1)
wordDFTest["AllUpperCase"] = wordDFTest.apply(lambda x: x["value"].isupper(), axis = 1)


# In[45]:


# creating feature 3
def checkFirstElement(row):
    if row["value"]:
        return row["value"][0].isupper()
    else:
        return False
wordDFTrain["FirstUpperCase"] = wordDFTrain.apply(lambda x: checkFirstElement(x), axis = 1)
wordDFTest["FirstUpperCase"] = wordDFTest.apply(lambda x: checkFirstElement(x), axis = 1)


# In[46]:


# creating feature 4: 
wordDFTrain["FirstUpperCase"] = wordDFTrain.apply(lambda x: len(x["value"]), axis = 1)
wordDFTest["FirstUpperCase"] = wordDFTest.apply(lambda x: len(x["value"]), axis = 1)


# In[47]:


# creating feature 5: 
wordDFTrain["AllNos"] = wordDFTrain.apply(lambda x: x["value"].isdigit(), axis = 1)
wordDFTest["AllNos"] = wordDFTest.apply(lambda x: x["value"].isdigit(), axis = 1)


# In[48]:


# creating feature 6 - If the left word was all Upper Case
import math
def checkLeftUpper(row):
    if row["LeftUpperCase"] and isinstance(row["LeftUpperCase"], str) :
        if row["LeftUpperCase"][-1] == ".":
            return False
        else:
            return row["LeftUpperCase"][0].isupper()
    else:
        return False
wordDFTrain["LeftUpperCase"] = wordDFTrain["tokens"].shift(1, axis = 0) 
wordDFTest["LeftUpperCase"] = wordDFTest["tokens"].shift(1, axis = 0) 
wordDFTrain["LeftUpperCase"] = wordDFTrain.apply(lambda x: checkLeftUpper(x), axis = 1)
wordDFTest["LeftUpperCase"] = wordDFTest.apply(lambda x: checkLeftUpper(x), axis = 1)


# In[49]:


# creating feature 7 - If the token is a stopword
import nltk
from nltk.corpus import stopwords

def checkIfStopWord(word):
    stopWordSet = set(stopwords.words('english'))
    if word["value"] in stopWordSet:
        return True
    else:
        return False
wordDFTrain["IfStopWord"] = wordDFTrain.apply(lambda x: checkIfStopWord(x), axis = 1)
wordDFTest["IfStopWord"] = wordDFTest.apply(lambda x: checkIfStopWord(x), axis = 1)


# In[50]:


# creating feature 8 - If the toekn is end of sentence
def checkEOS(row):
    if row["tokens"] and isinstance(row["tokens"], str) :
        if row["tokens"][-1] == ".":
            return True
    return False
wordDFTrain["IfEOS"] = wordDFTrain.apply(lambda x: checkEOS(x), axis = 1)
wordDFTest["IfEOS"] = wordDFTest.apply(lambda x: checkEOS(x), axis = 1)


# In[51]:


# creating the x and y set for training
y_train = wordDFTrain.labels
x_train = wordDFTrain.drop(["tokens", "labels"], axis = 1)


# In[52]:


# creating the x and y set for testing
y_test = wordDFTest.labels
x_test = wordDFTest.drop(["tokens", "labels"], axis = 1)


# In[53]:


import sklearn
from sklearn.tree import DecisionTreeClassifier 
from sklearn.preprocessing import LabelEncoder


# In[54]:


import numpy as np
x_train = x_train.fillna(value="-")
y_train = y_train.fillna(value="-")


# In[55]:


# encoding the values
le = LabelEncoder()
x_train["value"] = x_train["value"].astype(str)
x_train["value"] = le.fit_transform(x_train["value"])


# In[56]:


le = LabelEncoder()
x_test["value"] = x_test["value"].astype(str)
x_test["value"] = le.fit_transform(x_test["value"])


# In[57]:


Clf = DecisionTreeClassifier(criterion = "entropy")
Clf = Clf.fit(x_train, y_train)
y_pred = Clf.predict(x_test)


# In[58]:


# y_pred = y_pred.fillna(value="-")
y_test = y_test.fillna(value="-")
# y_test


# In[59]:


import sklearn.metrics as metrics
print("Accuracy for the 8 features:",metrics.accuracy_score(y_test, y_pred)*100)

