import numpy as np


trainingData = open("usps.train", "r")

data = [word
            for line in trainingData
            for word in line.split()]   #creating large Array out of Text

trainingData.close()

#variables
K = int(data[0]) #number of classes
D = int(data[1]) #number of dimensions
N = [0] * K #number of observations of every class k
#k = 0 #class of following obervation vector
sigma = [0] * D # we "simulate" diagonal matrix
means = [[0]*K]*D

#functions
def setNumberOfObservations():
    global numberOfAllObservations
    numberOfAllObservations = 0
    while numberOfAllObservations < np.divide(len(data)-2, 257):
        if int(data[numberOfAllObservations * 257 + 2]) == 10:     #warum muss 10 0 sein?!?!
            N[0] += 1
        else:
            N[int(data[numberOfAllObservations * 257 + 2])] += 1 #starting in line 2
        numberOfAllObservations += 1


def mean(k):
    meanX = [0] * D   # sum of all obervations labled with class k
    observationIterator = 0 #start at first observation

    if k == 0:
        k = 10

    while observationIterator < numberOfAllObservations: #summing up
        if int(data[observationIterator * 257 + 2]) == k:  #obervation found which is labled with k
            currentObservation = observationIterator * 257 + 2 +1
            for elementIterator in range(0, D):  #going over all D elements of obervation and sum on x
                meanX[elementIterator] += int(data[currentObservation + elementIterator])
        observationIterator += 1

    if k == 10:
        k = 0

    return [np.divide(i, N[k]) for i in meanX]

#def pooledCovarianceVector():
#    return [pooledVariance(d) for d in range(0,D)]


def pooledCovarianceVector():
    varianceX = [0] * D
    for observationIterator in range(0, numberOfAllObservations):
        currentClass = int(data[observationIterator * 257 + 2])
        if currentClass == 10:
            currentClass = 0
        currentObservation = observationIterator * 257 + 3
        for elementIterator in range(0, D):
            tempX = int(data[currentObservation + elementIterator])
            tempMean = means[currentClass][elementIterator]
            varianceX[elementIterator] += (tempX - tempMean)**2

    sigma = [np.divide(x, numberOfAllObservations) for x in varianceX]
    return sigma

def p(k):
    return np.divide(N[k], numberOfAllObservations)




###########CALCULATIONS###############
setNumberOfObservations()

for k in range(0, K):   #calculate the means
    means[k] = mean(k)

#############OUTPUT TO PARAMETER#################
out = open("usps d.param", "w")


out.write("d \n")
out.write(str(K))
out.write("\n")
out.write((str(D)))
out.write("\n")

for k in range(0,K):
    out.write(str(k))
    out.write("\n")
    out.write(str(p(k)))
    out.write("\n")
    for m in means[k]:
        out.write(str(m))
        out.write(" ")
    out.write("\n")

    for v in pooledCovarianceVector():
        out.write(str(v))
        out.write(" ")
    out.write("\n")
out.close()

