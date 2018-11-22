###################CLASSIFIER##########################
import numpy as np

testData = open("usps.test", "r")

test = [word
            for line in testData
            for word in line.split()]   #creating large Array out of Text

testData.close()

parameterData = open("usps d.param", "r")

parameter = [word
            for line in parameterData
            for word in line.split()]   #creating large Array out of Text

parameterData.close()

K = int(parameter[1])
D = int(parameter[2])
p = [float(parameter[i * (2 * D + 2) +4]) for i in range(0,K)]  #array of class posterior possebilities
variance = [float(parameter[256 + 5 + i]) for i in range(0,D)]
#M = [[0]*K]*K
M = [0] * (K * K)

def meanElement(k, d):
    return float(parameter[k * (2 * D + 2) +5 + d])


def classify(x):
    choosenClass = 0
    choosenClassProbability = 0
    for k in range(0, K):
        #calculating the sum and stuff
        currentClassProbability = 0
        for d in range(0, D):   #sum up
            currentClassProbability += np.divide((x[d] - meanElement(k, d))**2, variance[d])
        currentClassProbability = np.exp(-currentClassProbability)
        currentClassProbability *= p[k]
        #check if smaller then choosenClass
        if currentClassProbability > choosenClassProbability:   #distance smaller!
            choosenClassProbability = currentClassProbability
            choosenClass = k
    return choosenClass

def main():
    numberOfEvents = int(np.divide(len(test) - 2, D + 1))
    numberOfWrongClassifications = 0
    for eventIterator in range(0, numberOfEvents):
        realClass = int(test[eventIterator * (D + 1) + 2])
        if realClass == 10:
            realClass = 0
        observation = [int(test[eventIterator * (D + 1) + 3 + i]) for i in range(0, D)]
        assignedClass = classify(observation)
        print(assignedClass)
        print(realClass)
        #M[realClass][assignedClass] += 1
        M[realClass * 10 + assignedClass] += 1
        if assignedClass != realClass:
            numberOfWrongClassifications += 1


    errorRateFile = open("usps_d.error", "w")
    errorRateFile.write(str(np.divide(numberOfWrongClassifications, numberOfEvents)))
    errorRateFile.close()

    writeConfusionMatrix()


def writeConfusionMatrix():
    cm = open("usps_d.cm", "w")
    for y in range(0, K):
        for x in range(0, K):
            #cm.write(str(M[y][x]))
            cm.write(str(M[y * 10 + x]))
            cm.write(" ")
        cm.write("\n")
    cm.close()

main()

