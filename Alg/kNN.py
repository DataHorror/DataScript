from __future__ import division
from numpy import *
import operator

def file2matrix(filename):
    fr=open(filename)
    arrayOfLines=fr.readlines()
    numberOfLines=len(arrayOfLines)
    numberOfRows=len(arrayOfLines[0].strip().split('\t'))
    returnMat=zeros((numberOfLines,numberOfRows-1))
    classVector=[]
    index=0
    for line in arrayOfLines:
        line=line.strip()
        listOfLine=line.split('\t')
        returnMat[index,:]=listOfLine[0:3]
        classVector.append(listOfLine[-1])
        index+=1
    return returnMat,classVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

def classfy(inX,filename,k=10):
    dataSet,labels=file2matrix(filename)
    dataSet,ranges,minVals=autoNorm(dataSet)
    Minus=tile(inX,(dataSet.shape[0],1))-dataSet
    a=Minus**2
    distance=a.sum(axis=1)**0.5
    sortedDistIndices=distance.argsort()
    classCount={}
    for i in range(k):
        votelabel=labels[sortedDistIndices[i]]
        classCount[votelabel]=classCount.get(votelabel,0)+1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def datingClassTest():
    hoRatio=0.1
    datingDataMat,datingLabels=file2matrix('datingTestSet.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        