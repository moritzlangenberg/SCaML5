###################CLASSIFIER##########################
import numpy as np
from numpy.linalg import inv

#testData = open("usps.test", "r")
test = np.fromfile("usps.test", dtype=float, count=-1, sep=" ")


dataPD = open("usps_pd.param", "r")

parameterPD = [word
            for line in dataPD
            for word in line.split()]  #creating large Array out of Text

dataPD.close()

parameterData = open("usps_pf.param", "r")

parameterPF = [word
            for line in parameterData
            for word in line.split()]   #creating large Array out of Text

parameterData.close()

parameterData = open("usps_cd.param", "r")

parameterCD = [word
            for line in parameterData
            for word in line.split()]   #creating large Array out of Text

parameterData.close()

parameterData = open("usps_cf.param", "r")

parameterCF = [word
            for line in parameterData
            for word in line.split()]   #creating large Array out of Text

parameterData.close()

paramPD = np.asarray(np.delete(parameterPD, 0), dtype=float)
paramPF = np.asarray(np.delete(parameterPF, 0), dtype=float)
paramCD = np.asarray(np.delete(parameterCD, 0), dtype=float)
paramCF = np.asarray(np.delete(parameterCF, 0), dtype=float)

#parsing is done here
def getObservationVector(n):
    return test[3+257*n: 259 + 257*n]

def getObservationClass(n):
    return int(test[2 + n*257])

def getMean(k):
    return paramPD[4+((256+258)*k): 256+4+(256+258)*k]

def getPD():
    return np.diag(paramPD[260: (260+256)])

def getPF():
    return np.array(paramPF[260: 65796]).reshape((256, 256))

def getCD(k):
    return np.diag(paramCD[260 + (256+256+2)*k: (260+256) + (256+256+2)*k])

def getCF(k):
    return np.array(paramPF[260 +(256+256*256+2)*k: 65796 + (256+256*256+2)*k]).reshape((256, 256))

#tests
#print(getObservationVector(2))
#print(getObservationClass(0))
#print(getObservationClass(1))
#print(getMean(0))
#print(getMean(1))
#print(getCF(0))
#print(getPF())
#print(getCD(1))

K = int(parameterPD[1])
D = int(parameterPD[2])
p = [float(parameterPD[i * (2 * D + 2) + 4]) for i in range(0, K)]  #array of class posterior possebilities
pooledDiagonalCovariance = np.array(float(parameterPD[256 + 5 + i]) for i in range(0, D))
pooledFullCovariance = np.array([float(parameterPF[256 + 5 + i + j]) for i in range(0, D)] for j in range(0, D))
classSpecificDiagonalCovariance = np.array(float(parameterCD[256 + 5 + i]) for i in range(0, D))
ClassSpecificFullCovariance = np.array(float(parameterCF[256 + 5 + i]) for i in range(0, D))

#M = [[0]*K]*K
M = [0] * (K * K)
lamda = [1, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

#def meanElement(k, d):
 #   return float(parameter[k * (2 * D + 2) +5 + d])


def classifyPD(n):
    choosenClass = 0
    choosenClassProbability = 0
    for k in range(0, K):
        currentClassProbability = p[k] * np.exp(-0.5 *
                                                np.sum(
                                                    (np.dot(getObservationVector(n) - getMean(k),
                                                    np.linalg.inv(getPD())))
                                                    * (getObservationVector(n) - getMean(k))))
        #check if smaller then choosenClass
        if currentClassProbability > choosenClassProbability:   #distance smaller!
            choosenClassProbability = currentClassProbability
            choosenClass = k
    return choosenClass

def classifyPF(n):
    choosenClass = 0
    choosenClassProbability = 0
    for k in range(0, K):
        currentClassProbability = p[k] * np.exp(-0.5 *
                                                np.sum(
                                                    (np.dot(getObservationVector(n) - getMean(k),
                                                    np.linalg.inv(getPF())))
                                                    * (getObservationVector(n) - getMean(k))))
        #check if smaller then choosenClass
        if currentClassProbability > choosenClassProbability:   #distance smaller!
            choosenClassProbability = currentClassProbability
            choosenClass = k
    return choosenClass

def classifyCD(n, l):
    choosenClass = 0
    choosenClassProbability = 0
    for k in range(0, K):
        currentClassProbability = p[k] * np.exp(-0.5 *
                                                np.sum(
                                                    (np.dot(getObservationVector(n) - getMean(k),
                                                     np.linalg.inv((l * getPD()) + ((1-l) * getCD(k)))))
                                                    * (getObservationVector(n) - getMean(k))))
        #check if smaller then choosenClass
        if currentClassProbability > choosenClassProbability:   #distance smaller!
            choosenClassProbability = currentClassProbability
            choosenClass = k
    return choosenClass

def classifyCF(n, l):
    choosenClass = 0
    choosenClassProbability = 0
    for k in range(0, K):
        currentClassProbability = p[k] * np.exp(-0.5 *
                                                np.sum(
                                                    (np.dot(getObservationVector(n) - getMean(k),
                                                    np.linalg.inv((l * getPF()) + ((1-l)*getCF(k)))))
                                                    * (getObservationVector(n) - getMean(k))))
        #check if smaller then choosenClass
        if currentClassProbability > choosenClassProbability:   #distance smaller!
            choosenClassProbability = currentClassProbability
            choosenClass = k
    return choosenClass




def main():
    #TESTS
    #print(getObservationClass(1))  works
    #print( getObservationVector(1)) works
    #print(getMean(getObservationClass(1)))  works now
    #print(getObservationVector(1)- getMean(getObservationClass(1))) works
    #print(np.array([getObservationVector(1)-getMean(getObservationClass(1))]).T) works
    #print((np.array([getObservationVector(1)-getMean(getObservationClass(1))]).T) @ (np.array([getObservationVector(1)-getMean(getObservationClass(1))]).T))
    #print(getObservationVector(1) - getMean(getObservationClass(1)) @ (np.array([getObservationVector(1) - getMean(getObservationClass(1))]).T))
    #print(np.dot((getObservationVector(1)-getMean(getObservationClass(1))), (getObservationVector(1)-getMean(getObservationClass(1))).T))
    #print(np.sum((np.dot(getObservationVector(1)-getMean(getObservationClass(1)), np.linalg.inv(getPD()))) * (getObservationVector(1)-getMean(getObservationClass(1)))))
    #print(np.sum((np.dot(getObservationVector(n) - getMean(getObservationClass(n)), np.linalg.inv(getPD()))) * (getObservationVector(n) - getMean(getObservationClass(n)))))
    #print((getObservationVector(1) - getMean(getObservationClass(1))) * (getObservationVector(1) - getMean(getObservationClass(1))))
    #print((0.5 * getPD()) + ((1-0.5) * getCD(1)))
    #print(getCD(0))

    numberOfEvents = int(np.divide(len(test) - 2, D + 1))
    numberOfWrongClassificationsCF = [0] * len(lamda)
    numberOfWrongClassificationsCD = [0] * len(lamda)
    numberOfWrongClassificationsPF = 0
    numberOfWrongClassificationsPD = 0

    for n in range(0, numberOfEvents):
        realClass = getObservationClass(n)
        if realClass == 10:
            realClass = 0
        assignedClassPD = classifyPD(n)
        assignedClassPF = classifyPF(n)
        assignedClassCD = [classifyCD(n, l) for l in lamda]
        assignedClassCF = [classifyCF(n, l) for l in lamda]

        if assignedClassPD != realClass:
            numberOfWrongClassificationsPD += 1

        if assignedClassPF != realClass:
            numberOfWrongClassificationsPF += 1

        for i in range(0, len(lamda)):
            if assignedClassCF[i] != realClass:
                numberOfWrongClassificationsCF[i] += 1

        for i in range(0, len(lamda)):
            if assignedClassCD[i] != realClass:
                numberOfWrongClassificationsCD[i] += 1




    ##ERRORFILE ORDER
    #PD
    #PF
    #CD all lamdas
    #CF all lamdas
    errorRateFile = open("usps_d.error", "w")
    errorRateFile.write(str(np.divide(numberOfWrongClassificationsPD, numberOfEvents)))
    errorRateFile.write("\n")
    errorRateFile.write(str(np.divide(numberOfWrongClassificationsPF, numberOfEvents)))
    errorRateFile.write("\n")
    for l in range(0, len(lamda)):
        errorRateFile.write(str(np.divide(numberOfWrongClassificationsCD[l], numberOfEvents)))
        errorRateFile.write("\t")
    errorRateFile.write("\n")
    for l in range(0, len(lamda)):
        errorRateFile.write(str(np.divide(numberOfWrongClassificationsCF[l], numberOfEvents)))
        errorRateFile.write("\t")
    errorRateFile.write("\n")
    errorRateFile.close()

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
