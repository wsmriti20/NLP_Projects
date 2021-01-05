#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import re
from collections import Counter
import string
import io


# In[26]:


preText = ''
with io.open("DialogAct.train", mode = 'r', encoding = "utf-8") as file:
    preText = file.read()
inputData = preText.strip().split("\n\n")
df = pd.DataFrame(inputData)
inputDataDF = df.apply (lambda x: x[0].strip().split("\n"), axis = 1)
inputDataList  = inputDataDF.to_list()
# inputDataDF[0]


# In[27]:



# len(inputDataList)


# In[28]:


# processing the input data of dialogues to pair DialogueAct with students message
# pair = (DialogAct, Message)
# inputDataDF
def processingPair(inputDialog):
    newDialog = []
    for line in inputDialog:
        lineList = line.split(":")
        agent = lineList[0]
        # print(agent)
        elementList = []
        if agent == "Advisor":
            w = re.search('\[(.*)\]', lineList[1])
            actLabel = w.group(1)
            w = re.search('\](.*)', lineList[1])
            sentence = w.group(1).strip()
            # elementList.append(agent)
            elementList.append(actLabel)
            elementList.append(sentence)
        else:
            sentence = lineList[1].strip()
            # elementList.append(agent)
            elementList.append(None)
            elementList.append(sentence)
        newDialog.append(elementList)
    return newDialog

# df[0]


# In[29]:


# pairing the labels with previous Dialog
from string import punctuation
def collectingTheAppropriatePair(inputDialog):
    newPairList = []
    if inputDialog:
        if inputDialog[0][0]:
            newPairList.append((inputDialog[0][0], None))
        for i in range(1, len(inputDialog)):
            if inputDialog[i][0]:
                wordList = inputDialog[i-1][1].split(" ")
                if wordList:
                    wordList = list(map(lambda x: x.strip(punctuation), wordList))
                newPair = (inputDialog[i][0], wordList)
                newPairList.append(newPair)
        return newPairList
# collectingTheAppropriatePair(df[0])


# In[30]:


# processing the bag of words
# inputDataList[0][3]
def creatingBagOfWords(instanceRawText, senseWordCountDict, sensesDict, wordSet):
    sense = instanceRawText[0]
    # print(sensesDict)
    sensesDict[sense] += 1
    listOfWords = instanceRawText[1]
    # print(listOfWords)
    if listOfWords:
        for word in listOfWords:
            wordSet.add(word)
            if word in senseWordCountDict[sense].keys():
                senseWordCountDict[sense][word] += 1
            else:
                senseWordCountDict[sense][word] = 1   
    return senseWordCountDict
# print(creatingBagOfWords(inputDataList[0], senseWordsDict))
# inputDataList[0]


# In[31]:


df = inputDataDF.apply(lambda x: processingPair(x))
df = df.apply(lambda x: collectingTheAppropriatePair(x))
pairedList = df.to_list()

inputDataList = []
for i in pairedList:
    if i:
        for e in i:
            inputDataList.append(e)
# pairedList = [item for sublist in pairedList for item in sublist]
# inputDataList[0]
# pairedList[:5]
# df.head(5)
# Processing senses and counts
sensesList = list(map(lambda x: x[0], inputDataList))
# sensesList
uniqueSenses = list(set(sensesList))
# uniqueSenses
sensesCountDict = dict(Counter(sensesList))
# sensesCountDict
senseWordsDict = {key: {} for key in uniqueSenses} 
# senseWordsDict


# In[32]:


# applying the function on all the four trainings
# maybe remove the word we are looking into?
# remove stop words
# modify the training to rotate the folds
noTotalInstances = 0
sensesCountDict = dict.fromkeys((uniqueSenses), 0)
senseWordsDict = {key: {} for key in uniqueSenses} 
uniqueWordSet = set()

for instance in inputDataList:
    noTotalInstances += 1
    creatingBagOfWords(instance, senseWordsDict, sensesCountDict, uniqueWordSet)
# test = folds[4]
# senseWordsDict
# sensesCountDict
# len(uniqueWordSet)


# In[33]:


def calculateSenseProb (prob_sense, sense_count_dict, total):
    for sense in prob_sense.keys():
        prob_sense[sense] = float(sense_count_dict[sense] / total)


# In[34]:


def calculateWordSenseProb(probWordSenseDict, contextList, senseWord, v, sensesCount, uniqueSense):
    if contextList:
        for word in contextList:
            probWordSenseDict[word] = {}
            if uniqueSense:
                for sense in uniqueSense:
                    if word in senseWord[sense].keys():
                        probWordSenseDict[word][sense] = float((senseWord[sense][word] + 1)/(sensesCount[sense] + v))
                    else:
                        probWordSenseDict[word][sense] = float((1)/(sensesCount[sense] + v))


# In[35]:


def calculatingSenseForInstance(probSenseWordPred, contextWordList, senseList, probSense, probWordSense):
    if sensesList:
        for sense in senseList:
            if contextWordList:
                for word in contextWordList:
                    probSenseWordPred[sense] += math.log10(probWordSense[word][sense])
                probSenseWordPred[sense] += math.log10(probSense[sense])


# In[36]:


# doing the Naive Based Classification   
# doing the Naive Based Classification   
def naiveBasedClassifier(ListTesting, senseCountDict,senseWordsDict):
    # calculating the probablitiy of sense 
    # P(Sense)
    probSenseDict = dict.fromkeys((uniqueSenses), 0)
    calculateSenseProb(probSenseDict,senseCountDict,noTotalInstances)
    
    predictionInstances = []

    # find the probablity of words given sense, applying add one smoothing
    # P(feature|sense)]
    V = len(uniqueWordSet)
    probWordSenseDict = {}
    probSenseWordPred = dict.fromkeys((uniqueSenses), 0)
    for instance in ListTesting:
        contextWordList = instance[1]
        calculateWordSenseProb(probWordSenseDict, contextWordList, senseWordsDict, V, sensesCountDict, uniqueSenses)

    for instance in ListTesting:
        contextWordList = instance[1]
        # Calculating the sense given word context
        # print(contextTestList[0])
        calculatingSenseForInstance(probSenseWordPred, contextWordList, uniqueSenses, probSenseDict, probWordSenseDict)

        # finding max
        predSense = max(probSenseWordPred, key = probSenseWordPred.get)
        # storing the instances ID and prediction sense
        predictionInstances.append(predSense)

    return predictionInstances


# In[37]:


# prepping the test data
preText = ''
with io.open("DialogAct.test", mode = 'r', encoding = "utf-8") as file:
    preText = file.read()
inputTestData = preText.strip().split("\n\n")
dfTest = pd.DataFrame(inputTestData)
inputTestDataDF = dfTest.apply (lambda x: x[0].strip().split("\n"), axis = 1)
inputTestDataList  = inputTestDataDF.to_list()

dfTest = inputTestDataDF.apply(lambda x: processingPair(x))
dfTest = dfTest.apply(lambda x: collectingTheAppropriatePair(x))
pairedListTest = dfTest.to_list()

inputTestDataList = []
for i in pairedListTest:
    if i:
        for e in i:
            inputTestDataList.append(e)
# inputTestDataList[:4]
processedListTesting = inputTestDataList


# In[38]:


def accuracy(predList, actualList):
    correct = 0
    for i in range(len(predList)):
        if predList[i] == actualList[i][0]:
            correct += 1
    return (correct*100/len(predList))


# In[39]:


predictedTests = naiveBasedClassifier(processedListTesting, sensesCountDict, senseWordsDict)
# predictedTests


# In[40]:


accuracyValue = accuracy(predictedTests, processedListTesting)
accuracyValue

