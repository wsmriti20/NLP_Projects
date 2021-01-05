import pandas as pd
import re
from collections import Counter
import string
import math

# Creating the five folds:
def creatingFiveFolds(inputList):
    folds = []
    totalInstances = int(math.ceil(len(inputList)*0.1)/0.1)
    # if (totalInstances % 5 == 0):
    # if (totalInstances%5)
    indices = [ int(totalInstances/5 + i*totalInstances/5) for i in range(5)]
    folds.append(inputList[0: indices[0]])
    folds.append(inputList[indices[0]: indices[1]])
    folds.append(inputList[indices[1]: indices[2]])
    folds.append(inputList[indices[2]: indices[3]])
    folds.append(inputList[indices[3]: indices[4]])
    folds.append(inputList[indices[4]: ])
    return folds
	
# function to get sense from the raw instance to a df for senses
def extractingTheSense(instanceRawText):
    answerInstance = instanceRawText[1].split(" ")
    sense_reg1 = re.search('\"(.*)\"', answerInstance[2])
    sense1 = sense_reg1.group(1)
    sense_reg2 = re.search('%(.*)', sense1)
    sense2 = sense_reg2.group(1)
    return sense2

def extractingTheID(instanceRawText):
    answerInstance = instanceRawText[1].split(" ")
    ID_reg1 = re.search('\"(.*)\"', answerInstance[1])
    ID1 = ID_reg1.group(1)
    ID_reg2 = re.search('\.(.*)', ID1)
    ID2 = ID_reg2.group(1)
    return ID2

def extractingTheContext(instanceRawText):
    listOfWords = instanceRawText[3].split(" ")
    listOfWords = list(map(lambda x: re.sub(r'[^\w\s]','',x), listOfWords))
    listOfWords = list(filter(lambda x: x != '', listOfWords))
    return listOfWords

# processing the bag of words
def creatingBagOfWords(instanceRawText, senseWordCountDict, sensesDict, wordSet):
    sense = extractingTheSense(instanceRawText)
    # print(sensesDict)
    sensesDict[sense] += 1
    listOfWords = extractingTheContext(instanceRawText)
    
    for word in listOfWords:
        wordSet.add(word)
        if word in senseWordCountDict[sense].keys():
            senseWordCountDict[sense][word] += 1
        else:
            senseWordCountDict[sense][word] = 1   
    return senseWordCountDict

def calculateSenseProb (prob_sense, sense_count_dict, total):
    for sense in prob_sense.keys():
        prob_sense[sense] = float(sense_count_dict[sense] / total)

def calculateWordSenseProb(probWordSenseDict, contextList, senseWord, v, sensesCount, uniqueSense):
    for word in contextList:
        probWordSenseDict[word] = {}
        for sense in uniqueSense:
            if word in senseWord[sense].keys():
                probWordSenseDict[word][sense] = float((senseWord[sense][word] + 1)/(sensesCount[sense] + v))
            else:
                probWordSenseDict[word][sense] = float((1)/(sensesCount[sense] + v))

def calculatingSenseForInstance(probSenseWordPred, contextWordList, senseList, probSense, probWordSense):
    for sense in senseList:
        for word in contextWordList:
            probSenseWordPred[sense] += math.log10(probWordSense[word][sense])
        probSenseWordPred[sense] += math.log10(probSense[sense])


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
        contextWordList = instance[2]
        calculateWordSenseProb(probWordSenseDict, contextWordList, senseWordsDict, V, sensesCountDict, uniqueSenses)

    for instance in ListTesting:
        contextWordList = instance[2]
        # Calculating the sense given word context
        # print(contextTestList[0])
        calculatingSenseForInstance(probSenseWordPred, contextWordList, uniqueSenses, probSenseDict, probWordSenseDict)

        # finding max
        predSense = max(probSenseWordPred, key = probSenseWordPred.get)
        # storing the instances ID and prediction sense
        predictionInstances.append((instance[0], predSense))

    return predictionInstances

def accuracy(predList, actualList):
    correct = 0
    for i in range(len(predList)):
        if predList[i][1] == actualList[i][1]:
            correct += 1
    return (correct*100/len(predList))

preText = ''
with open('plant.wsd', 'r') as file:
    preText = file.read()
inputData = preText.strip().split("\n\n")
df = pd.DataFrame(inputData)
inputDataDF = df.apply (lambda x: x[0].strip().split("\n"), axis = 1)

f = open("plant.wsd.out", "a")

inputDataList  = inputDataDF.to_list()
len(inputDataList)

noInstances = len(inputDataDF)
noInstances

# Processing senses and counts
sensesList = list(map(lambda x: extractingTheSense(x), inputDataList))
uniqueSenses = list(set(sensesList))
sensesCountDict = dict(Counter(sensesList))
sensesCountDict
senseWordsDict = {key: {} for key in uniqueSenses} 

# applying the function on all the four trainings
# maybe remove the word we are looking into?
# remove stop words
# modify the training to rotate the folds
folds = creatingFiveFolds(inputDataList)
accuracyList = []
for i in range(5):
    f.write("Fold {:d}\n".format(i))
    test = folds[i]
    folds.pop(i)
    train = folds

    noTotalInstances = 0
    sensesCountDict = dict.fromkeys((uniqueSenses), 0)
    senseWordsDict = {key: {} for key in uniqueSenses} 
    uniqueWordSet = set()
    for fold in train:
        for instance in fold:
            noTotalInstances += 1
            creatingBagOfWords(instance, senseWordsDict, sensesCountDict, uniqueWordSet)

    # working in the test fold
    # I need the ID, Context (list of words)
    # Id and the gold standard senses
    sensesTestList = list(map(lambda x: extractingTheSense(x), test))
    IDTestList = list(map(lambda x: extractingTheID(x), test))
    contextTestList  = list(map(lambda x: extractingTheContext(x), test))
    processedListTesting = list(zip(IDTestList, sensesTestList, contextTestList))

    predictedTests = naiveBasedClassifier(processedListTesting, sensesCountDict, senseWordsDict)
    accuracyValue = accuracy(predictedTests, processedListTesting)
    print("Fold " + "{:d}".format(i+1) +  " accuracy: " + "{:.2f}".format(accuracyValue))
    accuracyList.append(accuracyValue)

    for element in predictedTests:
        f.write("plant.{:s}".format(element[0]) + " plant%{:s}".format(element[1]) + "\n")

    folds.append(test)
f.close()
print("Average accuracy: " + '{:.2f}'.format(sum(accuracyList)/5))
