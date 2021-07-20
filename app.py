from flask import Flask
from flask import Flask, render_template
from flask import request
import pandas as pd
import numpy as np
import scipy
from statistics import mode
from math import sqrt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import imblearn
from imblearn.over_sampling import SMOTE
import pengujian as p1
import time
from flask_paginate import Pagination, get_page_args, get_page_parameter

app = Flask(__name__)

# Fungsi untuk mendapatkan nilai min dan max
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

# Fungsi Manual MinMax Scaler
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



# Fungsi Prediksi KNN
def predictknn(point, data, label, n):
  distances = []
  for i in range(len(data)):
    distances.append([p1.euclidean(point, data[i]), label[i]])
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

# Fungsi KNN Manual
def diagnosis(a):
    #Import from csv
    d = pd.read_csv('static/fetal_health.csv')
    X = d.drop(columns="fetal_health")
    y = d.fetal_health
    y = np.array(y)

    # Normalisasi Data
    X_scaled = MinMaxScalerManual(np.array(X),np.array(X))

    # Balancing DATA SMOTE
    oversample = SMOTE()
    X_scaled, y = oversample.fit_resample(X_scaled, y)

    # Data Test
    data = [120.0,0.0,0.0,0.0,0.0,0.0,0.0,73.0,0.5,43.0,2.4,64.0,62.0,126.0,2.0,0.0,120.0,137.0,121.0,73.0,1.0]
    data = np.array(a).reshape(1, 21)
    data = MinMaxScalerManual(np.array(X),data)

    # Melakukan Prediksi dengan KNN
    prediksi = predictknn(data, X_scaled, y, 4)
    
    # Mengolah hasil Prediksi
    hasil = ""
    if(prediksi == 1):
        hasil = "Normal"
    elif(prediksi == 2):
        hasil = "Suspect"
    else:
        hasil = "Pathological"

    return hasil

@app.route("/")
def main():
    return render_template('index.html')

@app.route("/index2")
def index2():
    return render_template('index2.html')

@app.route("/result", methods=["POST"])
def result():
    
    # Start Time
    start_time = time.time()
    
    data = []
    data.append(float(request.form.get("baseline_value"))) #data0
    data.append(float(request.form.get("accelerations"))) #data1
    data.append(float(request.form.get("fetal_movement"))) #data2
    data.append(float(request.form.get("uterine_contractions"))) #data3
    data.append(float(request.form.get("light_decelerations"))) #data4
    data.append(float(request.form.get("severe_decelerations"))) #data5
    data.append(float(request.form.get("prolongued_decelerations"))) #data6
    data.append(float(request.form.get("abnormal_short_term_variability"))) #data7
    data.append(float(request.form.get("mean_value_of_short_term_variability"))) #data8
    data.append(float(request.form.get("percentage_of_time_with_abnormal"))) #data9
    data.append(float(request.form.get("mean_value_of_long_term_variability"))) #data10
    data.append(float(request.form.get("histogram_width"))) #data11
    data.append(float(request.form.get("histogram_min"))) #data12
    data.append(float(request.form.get("histogram_max"))) #data13
    data.append(float(request.form.get("histogram_number_of_peaks"))) #data14
    data.append(float(request.form.get("histogram_number_of_zeroes"))) #data15
    data.append(float(request.form.get("histogram_mode"))) #data16
    data.append(float(request.form.get("histogram_mean"))) #data17
    data.append(float(request.form.get("histogram_median"))) #data18
    data.append(float(request.form.get("histogram_variance"))) #data19
    data.append(float(request.form.get("histogram_tendency"))) #data20
    
    print(f"data: {data} ")
    hasil = diagnosis(data)

    # End Time
    end_time = time.time()
    time_result = end_time - start_time
    time_result = p1.time_convert(time_result)

    return render_template('result.html', data=data, hasil=hasil, time_result=time_result)

@app.route("/pengujian")
def pengujian():

  # Start Time Pengujian
  start_time_pengujian = time.time()

  #Import from csv
  d = pd.read_csv('static/fetal_health.csv')
  X = d.drop(columns="fetal_health")
  y = d.fetal_health
  y = np.array(y)

  # Normalisasi Data
  X_scaled = MinMaxScalerManual(np.array(X),np.array(X))

  # Balancing DATA SMOTE
  oversample = SMOTE()
  X_scaled, y = oversample.fit_resample(X_scaled, y)

  # Join Data dan Label
  y1 = y.reshape(4965, 1) 
  data = np.hstack((y1, X_scaled))

  kelas1 = p1.label1(data)
  kelas2 = p1.label2(data)
  kelas3 = p1.label3(data)

  knn = p1.pengujian1(kelas1,kelas2,kelas3)
  lenknn = len(knn[1])

  kfold = p1.pengujian2(kelas1,kelas2,kelas3)
  lenkfold = len(kfold[1])

  end_time_pengujian = time.time()
  time_pengujian = end_time_pengujian - start_time_pengujian
  time_pengujian = p1.time_convert(time_pengujian)
  
  return render_template('pengujian.html', knn=knn, lenknn=lenknn, kfold=kfold, lenkfold=lenkfold ,time_pengujian=time_pengujian)

#Import data asli / data awal from csv
data_asli = np.array(pd.read_csv('static/fetal_health.csv'))

def get_users_data_asli(offset=0, per_page=100):
    return data_asli[offset: offset + per_page]

@app.route("/datasetasli")
def datasetasli():

   #Import from csv
  # d = pd.read_csv('static/fetal_health.csv')
  # X = d.drop(columns="fetal_health")
  # y = d.fetal_health
  # y = np.array(y)
  # data = np.array(d)

  page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
  total = len(data_asli)
  pagination_users = get_users_data_asli(offset=offset, per_page=per_page)
  pagination = Pagination(page=page, per_page=per_page, total=total, css_framework='bootstrap4')
  return render_template('datasetasli.html', users=pagination_users, page=page, per_page=per_page, pagination=pagination,)

@app.route("/datasetasli2")
def datasetasli2():

  #Import from csv
  d = pd.read_csv('static/fetal_health.csv')
  X = d.drop(columns="fetal_health")
  y = d.fetal_health
  y = np.array(y)

  data = np.array(d)

  lendata = len(data) 
  return render_template('datasetasli2.html', data=data, lendata=lendata)


#Data SMOTE
# dataasli = np.array(pd.read_csv('static/fetal_health.csv'))
d_smote = pd.read_csv('static/fetal_health.csv')
X_asli = d_smote.drop(columns="fetal_health")
y_asli = np.array(d_smote.fetal_health)

#SMOTE
oversample = SMOTE()
X_smote, y_smote = oversample.fit_resample(np.array(X_asli), y_asli)

# Join Data dan Label
y_smote = y_smote.reshape(4965, 1) 
data_smote = np.hstack((X_smote, y_smote))
data_smote = np.round(data_smote, 3)

def get_users_data_smote(offset=0, per_page=100):
    return data_smote[offset: offset + per_page]
@app.route("/datasetsmote")
def datasetsmote():

  #Import from csv
  # d = pd.read_csv('static/fetal_health.csv')
  # X = d.drop(columns="fetal_health")
  # y = d.fetal_health
  # y = np.array(y)

  # oversample = SMOTE()
  # X_scaled, y = oversample.fit_resample(np.array(X), y)

  # Join Data dan Label
  # y1 = y.reshape(4965, 1) 
  # data = np.hstack((X_scaled, y1))

  page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
  total = len(data_smote)
  pagination_users = get_users_data_smote(offset=offset, per_page=per_page)
  pagination = Pagination(page=page, per_page=per_page, total=total, css_framework='bootstrap4')
  return render_template('datasetsmote.html', users=pagination_users, page=page, per_page=per_page, pagination=pagination,)

@app.route("/datasetsmote1")
def datasetsmote1():

  #Import from csv
  d = pd.read_csv('static/fetal_health.csv')
  X = d.drop(columns="fetal_health")
  y = d.fetal_health
  y = np.array(y)

  oversample = SMOTE()
  X_scaled, y = oversample.fit_resample(np.array(X), y)

  # Join Data dan Label
  y1 = y.reshape(4965, 1) 
  data = np.hstack((X_scaled, y1))

  lendata = len(data) 
  return render_template('datasetsmote.html', data=data, lendata=lendata)


# DATA AWAL


# def get_users(offset=0, per_page=100):
#     return data[offset: offset + per_page]



if __name__ == "__main__":
    app.run(debug=True)