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
sigmaPD = [0] * D # we "simulate" diagonal matrix
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
    meanX = np.zeros(D)   # sum of all obervations labled with class k
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


def pooledDiagonalCovariance():
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


################EXERCISE 4#######################
######numpy is not used because not explained anywhere in exercise and we are new to python


def classSpecificDiagonalCovariance():
    varianceX = np.zeros((D, K))
    classCount = [0] * K
    for observationIterator in range(0, numberOfAllObservations):
        currentClass = int(data[observationIterator * 257 + 2])
        if currentClass == 10:
            currentClass = 0
        classCount[currentClass] += 1
        currentObservation = observationIterator * 257 + 3
        for elementIterator in range(0, D):
            tempX = int(data[currentObservation + elementIterator])
            tempMean = means[currentClass][elementIterator]
            varianceX[elementIterator][currentClass] += (tempX - tempMean)**2

    for k in range(0, K):
        for x in range(0, K):
            varianceX[x][k] = np.divide(varianceX[x][k], classCount[k])
    return varianceX

def pooledFullCovariance():
    covarianceMatrix = np.zeros((D, D))
    for observationIterator in range(0, numberOfAllObservations):
        currentClass = int(data[observationIterator * 257 + 2])
        if currentClass == 10:
            currentClass = 0
        currentObservation = observationIterator * 257 + 3
        for elementIterator in range(0, D):
            tempX = int(data[currentObservation + elementIterator])
            tempMean = means[currentClass][elementIterator]
            for secondElementIterator in range(0, D):
                tempX2 = int(data[currentObservation + secondElementIterator])
                tempMean2 = means[currentClass][secondElementIterator]
                covarianceMatrix[elementIterator][secondElementIterator] += (tempX - tempMean)*(tempX2 - tempMean2)

   # for x in range(0,D):
    #    for y in range(0,D):
     #       covarianceMatrix[x][y] = np.divide(covarianceMatrix[x][y], numberOfAllObservations)

    return covarianceMatrix * (np.divide(1, numberOfAllObservations))

def classSpecificFullCovariance():
    covarianceMatrix = np.zeros((D, D, K))
    classCount = np.zeros(K)
    for observationIterator in range(0, numberOfAllObservations):
        currentClass = int(data[observationIterator * 257 + 2])
        if currentClass == 10:
            currentClass = 0
        classCount[currentClass] += 1
        currentObservation = observationIterator * 257 + 3
        for elementIterator in range(0, D):
            tempX = int(data[currentObservation + elementIterator])
            tempMean = means[currentClass][elementIterator]
            for secondElementIterator in range(0, D):
                tempX2 = int(data[currentObservation + secondElementIterator])
                tempMean2 = means[currentClass][secondElementIterator]
                covarianceMatrix[elementIterator][secondElementIterator][currentClass] += (tempX - tempMean)*(tempX2 - tempMean2)

    for x in range(0,D):
        for y in range(0,D):
            for k in range(0,K):
                covarianceMatrix[x][y][k] = np.divide(covarianceMatrix[x][y][k], classCount[k])

    return covarianceMatrix

##################################################
def p(k):
    return np.divide(N[k], numberOfAllObservations)




###########CALCULATIONS###############
setNumberOfObservations()

for k in range(0, K):   #calculate the means
    means[k] = mean(k)

#############OUTPUT TO PARAMETER#################
"""out = open("usps d.param", "w")


out.write("d \n")
out.write(str(K))
out.write("\n")
out.write((str(D)))
out.write("\n")

for k in range(0, K):
    out.write(str(k))
    out.write("\n")
    out.write(str(p(k)))
    out.write("\n")
    for m in means[k]:
        out.write(str(round(m, 1)))
        out.write("\t")
    out.write("\n")
    for v in pooledDiagonalCovariance():
        out.write(str(v))
        out.write("\t")
    out.write("\n")
out.close()
"""

#pd = pooledDiagonalCovariance()
#pf = pooledFullCovariance()
#cd = classSpecificDiagonalCovariance()
cf = classSpecificFullCovariance()

""""WORKS
out = open("usps_pd.param", "w")
out.write("d \n")
out.write(str(K))
out.write("\n")
out.write((str(D)))
out.write("\n")
for k in range(0, K):
    out.write(str(k))
    out.write("\n")
    out.write(str(p(k)))
    out.write("\n")
    for m in means[k]:
        out.write(str(m))
        out.write("\t")
    out.write("\n")
    for v in pd:
        out.write(str(v))
        out.write("\t")
    out.write("\n")
out.close()


out = open("usps_pf.param", "w")
out.write("f \n")
out.write(str(K))
out.write("\n")
out.write((str(D)))
out.write("\n")
for k in range(0, K):
    out.write(str(k))
    out.write("\n")
    out.write(str(p(k)))
    out.write("\n")
    for m in means[k]:
        out.write(str(m))
        out.write("\t")
    out.write("\n")
    for x in range(0, D):
        for y in range(0, D):
            out.write(str(pf[x][y]))
            out.write("\t")
        out.write("\n")
    out.write("\n")
out.close()


out = open("usps_cd.param", "w")
out.write("d \n")
out.write(str(K))
out.write("\n")
out.write((str(D)))
out.write("\n")
for k in range(0, K):
    out.write(str(k))
    out.write("\n")
    out.write(str(p(k)))
    out.write("\n")
    for m in means[k]:
        out.write(str(m))
        out.write("\t")
    out.write("\n")
    for y in range(0, D):
        out.write(str(cd[y][k]))
        out.write("\t")
    out.write("\n")
out.close()

"""

out = open("usps_cf.param", "w")
out.write("f \n")
out.write(str(K))
out.write("\n")
out.write((str(D)))
out.write("\n")
for k in range(0, K):
    out.write(str(k))
    out.write("\n")
    out.write(str(p(k)))
    out.write("\n")
    for m in means[k]:
        out.write(str(m))
        out.write("\t")
    out.write("\n")
    for y in range(0, D):
        for x in range(0, D):
            out.write(str(cf[x][y][k]))
            out.write("\t")
        out.write("\n")
    out.write("\n")
out.close()


