import pandas as pd
import numpy as np
from statistics import mode
from math import sqrt
import time

def time_convert(sec):
  mins = sec // 60
  sec = round(sec % 60, 3)
  hours = mins // 60
  mins = mins % 60
  return "{0}:{1}:{2}".format(int(hours),int(mins),sec)

# Fungsi Euclidean
def euclidean(r1, r2):
  return np.sqrt(np.sum((np.subtract(r1,r2))**2))

def knn(X_train, y_train, X_test, y_test, n):
  y_hat = [classify(point, X_train, y_train,n) for point in X_test]
  val = []
  for i in range(len(y_hat)):
    val.append(int(y_hat[i] == y_test[i]))
  return np.sum(val)/len(val)

def classify(point, data, label, n):
  distances = []
  for i in range(len(data)):
    distances.append([euclidean(point, data[i]), label[i]])
  distances.sort(key=lambda x: x[0])
  top_classes = [x[1] for x in distances[0:n]]
  # print(f"top_classes: { top_classes } ")
  repeat = True
  try:
    y_hat = mode(top_classes)
  except:
    while repeat:
      try:
        top_classes = top_classes[:-1]
        y_hat = mode(top_classes)
        repeat = False
      except:
        pass
        
  return y_hat

def label1(data):
  panjang = len(data)
  a = []
  for i in range(panjang):
    if(data[i][0] == 1):
      a.append(data[i][0:])
  return a
def label2(data):
  panjang = len(data)
  a = []
  for i in range(panjang):
    if(data[i][0] == 2):
      a.append(data[i][0:])
  return a
def label3(data):
  panjang = len(data)
  a = []
  for i in range(panjang):
    if(data[i][0] == 3):
      a.append(data[i][0:])
  return a

def kfcv(dataA, dataB, dataC, kfold, k):
  KA = int(len(dataA)/kfold)
  KB = int(len(dataB)/kfold)
  KC = int(len(dataC)/kfold)
  KA0 = 0
  KB0 = 0
  KC0 = 0
  hasilFold = []
  
  for i in range(kfold):
    klsTest = []
    klsTrain = []
    train = []
    test = []
    if i == kfold-1:
      KA0 = KAp
      KB0 = KBp
      KC0 = KBp
      KAp = len(dataA) 
      KBp = len(dataB) 
      KCp = len(dataC) 
    else:
      KA0 = KA*i
      KB0 = KB*i
      KC0 = KC*i
      KAp = KA*(i+1)
      KBp = KB*(i+1)
      KCp = KC*(i+1)

    for j in range(len(dataA)):
      if j>=KA0 and j<KAp :
        test.append(dataA[j][0:])
      else:
        train.append(dataA[j][0:])

    for j in range(len(dataB)):
      if j>=KB0 and j<KBp :
        test.append(dataB[j][0:])
      else:
        train.append(dataB[j][0:])

    for j in range(len(dataC)):
      if j>=KC0 and j<KCp :
        test.append(dataC[j][0:])
      else:
        train.append(dataC[j][0:])

    for j in range(len(train)):
      klsTrain.append(train[j][0])
    train = np.delete(train, [0,0], axis=1)

    for j in range(len(test)):
      klsTest.append(test[j][0])
    test = np.delete(test, [0,0], axis=1)

    #KNN manual
    akurasi = knn(train, klsTrain, test, klsTest, k)

    hasilFold.append(akurasi)

  return hasilFold

def pengujian1(kelas1,kelas2,kelas3):
    akurasi = []
    kterbaik = 0
    nilaikterbaik = 0
    terbaik = []
    hasil = []
    for i in range(3,23):
        start_time = time.time()
        a = []
        a.append(i)
        nilai = np.array(kfcv(kelas1,kelas2,kelas3,10, i)).mean()
        a.append(round(nilai, 3))
        end_time = time.time()
        time_lapsed = end_time - start_time
        time_lapsed = time_convert(time_lapsed)
        print(f"K: {a[0]} --> {a[1]}  ; time = {time_lapsed}")
        a.append(time_lapsed)
        akurasi.append(a)
        if(nilaikterbaik < a[1]):
            nilaikterbaik = a[1]
            kterbaik = a[0]
    terbaik.append(kterbaik)
    terbaik.append(round(nilaikterbaik, 3))
    hasil.append(terbaik)
    hasil.append(akurasi)
    return hasil

def pengujian2(kelas1,kelas2,kelas3):
    akurasi = []
    foldterbaik = 0
    nilaifoldterbaik = 0
    terbaik = []
    hasil = []
    for i in range(2,11):
        start_time = time.time()
        a = []
        a.append(i)
        nilai = np.array(kfcv(kelas1,kelas2,kelas3,i, 4)).mean()
        a.append(round(nilai, 3))
        end_time = time.time()
        time_lapsed = end_time - start_time
        time_lapsed = time_convert(time_lapsed)
        print(f"Kfold: {a[0]} --> {a[1]}  ; time = {time_lapsed}")
        a.append(time_lapsed)
        akurasi.append(a)
        if(nilaifoldterbaik < a[1]):
            nilaifoldterbaik = a[1]
            foldterbaik = a[0]
    terbaik.append(foldterbaik)
    terbaik.append(round(nilaifoldterbaik, 3))
    hasil.append(terbaik)
    hasil.append(akurasi)
    return hasil
