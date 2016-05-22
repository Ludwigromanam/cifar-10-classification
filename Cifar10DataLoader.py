import cPickle,os,urllib,subprocess,numpy as np

CIFAR_10_URL = "http://www.cs.utoronto.ca/~kriz/cifar-10-python.tar.gz"
CIFAR_10_DIRECTORY = "cifar-10-batches-py"

def downloadCifarData():
  print("Downloading CIFAR-10 data from " + CIFAR_10_URL)
  downloadFile = urllib.URLopener()
  downloadFile.retrieve(CIFAR_10_URL, "cifar-10-python.tar.gz")

  subprocess.call(["tar", "-zxvf", "cifar-10-python.tar.gz"])
  print("Extracted data into %s/"%CIFAR_10_DIRECTORY)


def unpickleBatch(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def fileForBatch(batchNum):
  return "%s/data_batch_%s"%(CIFAR_10_DIRECTORY, batchNum)

def toOneHot(data):
  v = data["labels"]
  oneHot = np.zeros([10000,10])
  oneHot[np.arange(10000), v] = 1
  data["labels"] = oneHot
  return data


def loadCifarDataForBatch(batchNum, oneHot = True):
  if (not os.path.isdir(CIFAR_10_DIRECTORY) or not os.path.exists(CIFAR_10_DIRECTORY)):
    downloadCifarData()
  data = unpickleBatch(fileForBatch(batchNum))
  return toOneHot(data) if oneHot else data

def loadTestBatch(oneHot = True):
  data = unpickleBatch("%s/test_batch"%(CIFAR_10_DIRECTORY))
  return toOneHot(data) if oneHot else data
