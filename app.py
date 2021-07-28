from flask import Flask
from flask import Flask, render_template, Response, send_file
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
# save numpy array as csv file
from numpy import asarray
from numpy import savetxt

app = Flask(__name__)

#FORM KLASIFIKASI

@app.route("/")
def main():
  return render_template('index.html')

#END FORM KLASIFIKASI

#GET DATA LATIH TERBAIK

@app.route("/data_latih")
def data_latih():

  #Import from csv
  d = pd.read_csv('static/fetal_health.csv')
  X = d.drop(columns="fetal_health")
  y = d.fetal_health
  y = np.array(y)

  # Normalisasi Data
  X_scaled = p1.MinMaxScalerManual(np.array(X),np.array(X))

  # Balancing DATA SMOTE
  oversample = SMOTE()
  X_scaled, y = oversample.fit_resample(X_scaled, y)

  # Join Data dan Label
  y1 = y.reshape(4965, 1) 
  data = np.hstack((y1, X_scaled))

  kelas1 = p1.label1(data)
  kelas2 = p1.label2(data)
  kelas3 = p1.label3(data)

  data1 = p1.getDataLatih(kelas1, kelas2, kelas3, 5)
  data_latih1 = np.array(data1[0]).round(4)
  # csv = '1,2,3\n4,5,6\n'
  download = np.savetxt("data_latih.csv", data_latih1, delimiter=",")

  download
  return render_template('index.html')
  #return Response(data_latih1,mimetype="text/csv",headers={"Content-disposition":"attachment; filename=data_latih.csv"})
  # return np.savetxt("data_latih.csv", data_latih1, delimiter=",")
  #return savetxt('data.csv', data_latih1, delimiter=',')
  #return send_file(data_latih1,mimetype="text/csv",attachment_filename="export.csv",

#END GET DATA LATIH TERBAIK

#PROSES KLASIFIKASI DATA BARU

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
  hasil = p1.diagnosis(data)

  # End Time
  end_time = time.time()
  time_result = end_time - start_time
  time_result = p1.time_convert(time_result)

  return render_template('result.html', data=data, hasil=hasil, time_result=time_result)

#END PROSES KLASIFIKASI DATA BARU

#PROSES PENGUJIAN

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
  X_scaled = p1.MinMaxScalerManual(np.array(X),np.array(X))

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

  kfold = p1.pengujian2(kelas1,kelas2,kelas3,10,4)
  lenkfold = len(kfold[1])

  end_time_pengujian = time.time()
  time_pengujian = end_time_pengujian - start_time_pengujian
  time_pengujian = p1.time_convert(time_pengujian)
  
  return render_template('pengujian.html', knn=knn, lenknn=lenknn, kfold=kfold, lenkfold=lenkfold ,time_pengujian=time_pengujian)

#END PROSES PENGUJIAN

#PROSES DATA AWAL

data_asli = np.array(pd.read_csv('static/fetal_health.csv'))
def get_users_data_asli(offset=0, per_page=100):
    return data_asli[offset: offset + per_page]
@app.route("/datasetasli")
def datasetasli():
  page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
  total = len(data_asli)
  pagination_users = get_users_data_asli(offset=offset, per_page=per_page)
  pagination = Pagination(page=page, per_page=per_page, total=total, css_framework='bootstrap4')
  return render_template('datasetasli.html', users=pagination_users, page=page, per_page=per_page, pagination=pagination,)

#END PROSES DATA AWAL

# PROSES SMOTE

d_smote = pd.read_csv('static/fetal_health.csv')
X_asli = d_smote.drop(columns="fetal_health")
y_asli = np.array(d_smote.fetal_health)
oversample = SMOTE()
X_smote, y_smote = oversample.fit_resample(np.array(X_asli), y_asli)
y_smote = y_smote.reshape(4965, 1) 
data_smote = np.hstack((X_smote, y_smote))
data_smote = np.round(data_smote, 3)
def get_users_data_smote(offset=0, per_page=100):
    return data_smote[offset: offset + per_page]
@app.route("/datasetsmote")
def datasetsmote():
  page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
  total = len(data_smote)
  pagination_users = get_users_data_smote(offset=offset, per_page=per_page)
  pagination = Pagination(page=page, per_page=per_page, total=total, css_framework='bootstrap4')
  return render_template('datasetsmote.html', users=pagination_users, page=page, per_page=per_page, pagination=pagination,)

#END PROSES SMOTE


# BEKAS BEKAS

@app.route("/index2")
def index2():
  return render_template('index2.html')

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
  return render_template('datasetsmote1.html', data=data, lendata=lendata)


if __name__ == "__main__":
    app.run(debug=True)