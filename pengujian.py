import pandas as pd
import numpy as np
from statistics import mode
from math import sqrt
import time
import scipy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import imblearn
from imblearn.over_sampling import SMOTE


# Variabel Global

dataset = pd.read_csv('static/fetal_health.csv')
X_dataset = dataset.drop(columns="fetal_health")
X_dataset = np.array(X_dataset)
y_dataset = dataset.fetal_health
y_dataset = np.array(y_dataset)

#TIMER
def time_convert(sec):
  mins = sec // 60
  sec = round(sec % 60, 1)
  hours = mins // 60
  mins = mins % 60
  if(hours == 0):
    if(mins == 0):
      return "{0} Detik".format(sec)
    else:
      return "{0} Menit {1} Detik".format(int(mins),sec)
  else:
    return "{0} Jam {1} Menit {2} Detik".format(int(hours),int(mins),sec)

#END TIMER

#EUCLIDEAN

def euclidean(r1, r2):
  return np.sqrt(np.sum((np.subtract(r1,r2))**2))

#END EUCLIDEAN

#KNN

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

#END KNN

# DIAGNOSIS

def diagnosis(a):
  y = []
  X = []
  #Import from csv
  d = pd.read_csv('static/data_latih_terbaik.csv', header=None)
  # X = d.drop(columns="fetal_health")
  # y = d.fetal_health
  # y = np.array(y)
  d = np.array(d)
  for j in range(len(d)):
    # print(f"data Train: {j}; ")
    y.append(d[j][0])
    X = np.delete(d, [0,0], axis=1)
  
  # Normalisasi Data
  # X_scaled = MinMaxScalerManual(np.array(X),np.array(X))

  # Balancing DATA SMOTE
  # oversample = SMOTE()
  # X_scaled, y = oversample.fit_resample(X_scaled, y)

  # Join Data dan Label
  # y1 = y.reshape(4965, 1) 
  # dataset = np.hstack((y1, X_scaled))

  # kelas1 = label1(dataset)
  # kelas2 = label2(dataset)
  # kelas3 = label3(dataset)

  # kelas_data_latih = []

  # data_latih = getDataLatih(kelas1, kelas2, kelas3, 5)
  data_uji = np.array(a).reshape(1, 21)
  data_uji = MinMaxScalerManual(X_dataset,data_uji)

  # Melakukan Prediksi dengan KNN
  prediksi = predictknn(data_uji, X, y, 4)
    
  # Mengolah hasil Prediksi
  hasil = ""
  if(prediksi == 1):
    hasil = "Normal"
  elif(prediksi == 2):
    hasil = "Suspect"
  else:
    hasil = "Pathological"

  return hasil

#END DIAGNOSIS

# PREDIKSI KNN

def predictknn(point, data, label, n):
  distances = []
  for i in range(len(data)):
    distances.append([euclidean(point, data[i]), label[i]])
  distances.sort(key=lambda x: x[0])
  top_classes = [x[1] for x in distances[0:n]]
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

#END PREDIKSI KNN

#NORMALISASI MINMAX

def MinMaxScalerManual(fit,b):
  minmax = getMinMax(fit)
  hasil = []
  for i in range(len(b)):
    baris = []
    for j in  range(len(b[i])):
      pembagi = minmax[j][1]-minmax[j][0]
      if (pembagi == 0):
        pembagi = 1
      c = (b[i][j] - minmax[j][0])/pembagi
      baris.append(c)
    hasil.append(baris)
  return hasil

#END NORMALISASI MANUAL

# GET NILAI MIN AND MAX

def getMinMax(x):
  minmax = []
  for i in range(0,21):
    kolom =[]
    min = x[0][i]
    max = x[0][i]
    for j in range(len(x)):
      if min > x[j][i]:
        min = x[j][i]
      if max < x[j][i]:
        max = x[j][i]
    kolom.append(min)
    kolom.append(max)
    minmax.append(kolom)
  return minmax

#END GET MINMAX

#PENGELOMPOKKAN PER LABEL / KELAS

#LABEL 1 / LABEL NORMAL
def label1(data):
  panjang = len(data)
  a = []
  for i in range(panjang):
    if(data[i][0] == 1):
      a.append(data[i][0:])
  return a

#LABEL 2 / LABEL SUSPECT
def label2(data):
  panjang = len(data)
  a = []
  for i in range(panjang):
    if(data[i][0] == 2):
      a.append(data[i][0:])
  return a

#LABEL 3 / LABEL PATHOLOGIC
def label3(data):
  panjang = len(data)
  a = []
  for i in range(panjang):
    if(data[i][0] == 3):
      a.append(data[i][0:])
  return a

#END PENGELOMPOKKAN DATA PER LABEL

#K-FOLD CROSS VALIDATION

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

#END K-FOLD CROSS VALIDATION

#PENGUJIAN 1 UNTUK MENDAPATKAN NILAI K TERBAIK

def pengujian1(kelas1,kelas2,kelas3):
    akurasi = []
    kterbaik = 0
    nilaikterbaik = 0
    terbaik = []
    hasil = []

    for i in range(3,5):
      if(i%2 == 0):
        start_time = time.time()
        a = []
        a.append(i)
        nilai = np.array(kfcv(kelas1,kelas2,kelas3,10, i)).mean()
        a.append(round(nilai, 4)*100)
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
    terbaik.append(round(nilaikterbaik, 4))
    hasil.append(terbaik)
    hasil.append(akurasi)
    return hasil

#END PENGUJIAN 1

#PENGUJIAN 2 UNTUK MENDAPATKAN FOLD TERBAIK DAN DATA LATIH TERBAIK

def pengujian2(dataA, dataB, dataC, Ka, nilaiK):
  KA = int(len(dataA)/Ka)
  KB = int(len(dataB)/Ka)
  KC = int(len(dataC)/Ka)
  KA0 = 0
  KB0 = 0
  KC0 = 0
  hasil_fold = []
  fold_terbaik = []
  nilai_fold = 0
  # data_train_terbaik = []
  # data_test_terbaik = []
  hasil_pengujian = []
  terbaik = []
  

  for i in range(Ka):
    a = []
    start_time = time.time()
    klsTest = []
    klsTrain = []
    train = []
    test = []
    data_train = []
    data_test = []
    # dataNormalTrain = 0
    # dataNormalTest = 0
    # dataSuspectTrain = 0
    # dataSuspectTest = 0
    # dataPathologicTrain = 0
    # dataPathologicTest = 0

    if i < 5 :
      if i == 0:
        KA0 = 0
        KB0 = 0
        KC0 = 0
        KAp = 165
        KBp = 166
        KCp = 166
      else:
        KA0 = KAp
        KB0 = KBp
        KC0 = KCp
        KAp += 165
        KBp += 166
        KCp += 166
    else :
      KA0 = KAp
      KB0 = KBp
      KC0 = KCp
      KAp += 166
      KBp += 165
      KCp += 165

    for j in range(len(dataA)):
      if j>=KA0 and j<KAp :
        data_test.append(dataA[j][0:])
        # dataNormalTest += 1
      else:
        data_train.append(dataA[j][0:])
        # dataNormalTrain += 1

    for j in range(len(dataB)):
      if j>=KB0 and j<KBp :
        data_test.append(dataB[j][0:])
        # dataSuspectTest += 1
      else:
        data_train.append(dataB[j][0:])
        # dataSuspectTrain += 1

    for j in range(len(dataC)):
      if j>=KC0 and j<KCp :
        data_test.append(dataC[j][0:])
        # dataPathologicTest += 1
      else:
        data_train.append(dataC[j][0:])
        # dataPathologicTrain += 1

    for j in range(len(data_train)):
      klsTrain.append(data_train[j][0])
    train = np.delete(data_train, [0,0], axis=1)

    for j in range(len(data_test)):
      klsTest.append(data_test[j][0])
    test = np.delete(data_test, [0,0], axis=1)

    a.append(i+1)
    #KNN
    akurasi = knn(train, klsTrain, test, klsTest, nilaiK)*100

    end_time = time.time()
    time_lapsed = end_time - start_time
    time_lapsed = time_convert(time_lapsed)
    a.append(akurasi)
    a.append(time_lapsed)

    hasil_fold.append(a)

    if nilai_fold < akurasi :
      fold_terbaik = i+1
      nilai_fold = akurasi
      # data_train_terbaik = data_train
      # data_test_terbaik = data_test
    print(f"Fold : {i+1}; Akurasi: {akurasi}; time: {time_lapsed}")

  terbaik.append(fold_terbaik)
  terbaik.append(nilai_fold)

  # print(f"Fold Terbaik: {fold_terbaik}; Akurasi: {nilai_fold}")
  hasil_pengujian.append(terbaik)
  hasil_pengujian.append(hasil_fold)
  
  return hasil_pengujian

# def pengujian2(kelas1,kelas2,kelas3):
#     akurasi = []
#     foldterbaik = 0
#     nilaifoldterbaik = 0
#     terbaik = []
#     hasil = []

#     for i in range(2,11):
#     # for i in range(2,5):
#       start_time = time.time()
#       a = []
#       a.append(i)
#       nilai = np.array(kfcv(kelas1,kelas2,kelas3,i, 4)).mean()
#       a.append(round(nilai, 4))
#       end_time = time.time()
#       time_lapsed = end_time - start_time
#       time_lapsed = time_convert(time_lapsed)
#       print(f"Kfold: {a[0]} --> {a[1]}  ; time = {time_lapsed}")
#       a.append(time_lapsed)
#       akurasi.append(a)
#       if(nilaifoldterbaik < a[1]):
#         nilaifoldterbaik = a[1]
#         foldterbaik = a[0]
#     terbaik.append(foldterbaik)
#     terbaik.append(round(nilaifoldterbaik, 4))
#     hasil.append(terbaik)
#     hasil.append(akurasi)
#     return hasil

#END PENGUJIAN 2


#GET DATA LATIH FOLD TERBAIK
def getDataLatih(dataA, dataB, dataC, Ka):
  KA0 = 0
  KB0 = 0
  KC0 = 0

  for i in range(Ka):
    if i < 5 :
      if i == 0:
        KA0 = 0
        KB0 = 0
        KC0 = 0
        KAp = 165
        KBp = 166
        KCp = 166
      else:
        KA0 = KAp
        KB0 = KBp
        KC0 = KCp
        KAp += 165
        KBp += 166
        KCp += 166
    else :
      KA0 = KAp
      KB0 = KBp
      KC0 = KCp
      KAp += 166
      KBp += 165
      KCp += 165

  #endfor

  data_train = []
  data_test = []
  dataNormalTrain = 0
  dataNormalTest = 0
  dataSuspectTrain = 0
  dataSuspectTest = 0
  dataPathologicTrain = 0
  dataPathologicTest = 0
  klsTrain = []
  klsTest = []
  
  for j in range(len(dataA)):
    if j>=KA0 and j<KAp :
      data_test.append(dataA[j][0:])
      dataNormalTest += 1
    else:
      data_train.append(dataA[j][0:])
      dataNormalTrain += 1

  for j in range(len(dataB)):
    if j>=KB0 and j<KBp :
      data_test.append(dataB[j][0:])
      dataSuspectTest += 1
    else:
      data_train.append(dataB[j][0:])
      dataSuspectTrain += 1

  for j in range(len(dataC)):
    if j>=KC0 and j<KCp :
      data_test.append(dataC[j][0:])
      dataPathologicTest += 1
    else:
      data_train.append(dataC[j][0:])
      dataPathologicTrain += 1

  for j in range(len(data_train)):
    # print(f"data Train: {j}; ")
    klsTrain.append(data_train[j][0])
    train = np.delete(data_train, [0,0], axis=1)

  # for j in range(len(data_test)):
  #     # print(f"datatest: {j}; ")
  #     klsTest.append(data_test[j][0])
  #     test = np.delete(data_test, [0,0], axis=1)

  # print(f"Fold: {i+1}; Data Latih: {len(data_train)} dan Data Uji: {len(data_test)}")
  # print(f"DataNormalTrain: {dataNormalTrain}; DataNormalTest: {dataNormalTest}")
  # print(f"DataSuspectTrain: {dataSuspectTrain}; DataSuspectTest: {dataSuspectTest}")
  # print(f"DataPathologicTrain: {dataPathologicTrain}; DataPathologicTest: {dataPathologicTest}")
  # print(f" ")


  return data_train,train,klsTrain