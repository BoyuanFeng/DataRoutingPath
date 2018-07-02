from scipy.stats.stats import pearsonr
import pickle
import statistics
import sys
arg1 = 'allGates1'


def sameClassCorrelation(allGates):
  correlationByLayer = []
  for layer in range(len(allGates[0])):
    layerCorrelation = []
    for i in range(len(allGates)):
      for j in range(len(allGates)):
        if j <= i:
          continue
        if allGates[i] == [] or allGates[j] == []:
          continue
        layerCorrelation.append(pearsonr(allGates[i][layer], allGates[j][layer])[0])
    correlationByLayer.append(sum(layerCorrelation)/len(layerCorrelation))
  print(correlationByLayer)




def twoClassesCorrelation(gatesOne, gatesTwo):
  correlationByLayer = []
  for layer in range(15):
    layerCorrelation = []
    for i in range(len(gatesOne)):
      for j in range(len(gatesTwo)):
        layerCorrelation.append(pearsonr(gatesOne[i][layer], gatesTwo[j][layer])[0])
    correlationByLayer.append(sum(layerCorrelation)/len(layerCorrelation))
  print(correlationByLayer)

def separate(allGates):
  separated = []
  for i in range(int(len(allGates)/10)):
    separated.append((allGates[i*10:(i+1)*10]))
  return separated


def allClasses(separated):
  for i in range(len(separated)):
    for j in range(len(separated)):
      if j <= i:
        continue
      twoClassesCorrelation(separated[i], separated[j])

def setToPositive(allGates):
  for i in range(len(allGates)):
    for j in range(len(allGates[i])):
      for k in range(len(allGates[i][j])):
        allGates[i][j][k] = abs(allGates[i][j][k])
  

def clip(allGates):
  for i in range(len(allGates)):
    for j in range(len(allGates[i])):
      for k in range(len(allGates[i][j])):
        if abs(allGates[i][j][k]) < 0.01:
          allGates[i][j][k] = 0


def count(x):
  count = 0
  for i in x:
    if i > 0.01:
      count += 1
  return count




arg0 = 'allGates0'
allGates0 = pickle.load(open(arg0,'rb'))
setToPositive(allGates0)
clip(allGates0)
sameClassCorrelation(allGates0)

arg1 = 'allGates1'
allGates1 = pickle.load(open(arg1,'rb'))
setToPositive(allGates1)
clip(allGates1)
sameClassCorrelation(allGates1)

arg2 = 'allGates2'
allGates2 = pickle.load(open(arg2,'rb'))
setToPositive(allGates2)
clip(allGates2)
sameClassCorrelation(allGates2)


twoClassesCorrelation(allGates0, allGates1)
twoClassesCorrelation(allGates0, allGates2)
twoClassesCorrelation(allGates1, allGates2)



def nonZeroPercent(x):
  for i in x:
    print("Non-zero: " + str(count(i)) + ", total: " + str(len(i)) + ", percent: "+str(count(i)*1.0/len(i)))


def statistic(allImages):
  for image in allImages:
    nonZeroPercent(image)



def absList(x):
  return [abs(i) for i in x]


def concat(x):
  result = []
  for i in x:
    result += i
  return result


def percentage(x, percentage):
  tmp = concat(x)
  idx = int(percentage*len(tmp))
  return sorted(absList(tmp))[idx]


def percentile(x, threshold):
  tmp = sorted(absList(layer))
  for i in range(len(tmp)):
    if tmp[i] > threshold:
      print((1.0*i)/len(x))
      break



def percentByLayer(x, threshold):
  for i in x:
    percentile(i, threshold)
    


def analyze(x, p):
  threshold = percentage(x, p)
  percentByLayer(x, threshold)







