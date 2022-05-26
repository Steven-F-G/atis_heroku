import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#from mapping import *

from ensurepip import bootstrap
from flask import Flask, request, redirect,url_for
app = Flask(__name__)

import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import pickle
from tqdm import tqdm
tqdm.pandas()
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Activation, Flatten, Dense, Conv1D, MaxPooling1D, Dropout
from keras.models import Model, load_model
from keras_self_attention import SeqSelfAttention
from sentence_transformers import SentenceTransformer
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
import nltk
from nltk import pos_tag
from nltk.tree import Tree
from nltk.chunk import conlltags2tree


import flask 
from flask import jsonify
from flask_bootstrap import Bootstrap

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from webdriver_manager.chrome import ChromeDriverManager
import re
from selenium import webdriver

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def charCNN_input_preprocess(text):
  input_text = np.array([text])
  input_text = [s.lower() for s in input_text]
  tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
  tk.fit_on_texts(input_text)

  alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
  char_dict = {}
  for i, char in enumerate(alphabet):
      char_dict[char] = i + 1

  tk.word_index = char_dict.copy()
  tk.word_index[tk.oov_token] = max(char_dict.values()) + 1

  input_sequence = tk.texts_to_sequences(input_text)
  input_data = pad_sequences(input_sequence, maxlen=1014, padding='post')
  input_data = np.array(input_data, dtype='float32')

  return input_data


def mapping_func(intent,dac,source,destination):
    if intent == "atis_flight" and dac == "question": #1 atis_flight
        return "The required flights from "+source+" to "+destination+" are.."
    elif intent == "atis_flight" and dac == "command":
        return "Please find below the requested flights from "+source+" to "+destination
    elif intent == "atis_flight" and dac == "statement":
        return "Sure.. Here are the requested flights from "+source+" to "+destination
    elif intent == "atis_airfare" and dac == "question": #2 atis_airfare
        return "The required air fares are.."
    elif intent == "atis_airfare" and dac == "command":
        return "Please find below the requested flight airfares"
    elif intent == "atis_airfare" and dac == "statement":
        return "Sure.. Here are the requested flight airfares"
    elif intent == "atis_flight_time" and dac == "question": #3 atis_flight_time
        return "The required flight timings for "+source+" to "+destination+" are.."
    elif intent == "atis_flight_time" and dac == "command":
        return "Please find below the flight timings for "+source+" to "+destination
    elif intent == "atis_flight_time" and dac == "statement":
        return "Sure.. Here are the flight timings for "+source+" to "+destination
    elif intent == "atis_airline" and dac == "question": #4 atis_airline
        return "The required airlines are.."
    elif intent == "atis_airline" and dac == "command":
        return "Please find below the requested flight airlines"
    elif intent == "atis_airline" and dac == "statement":
        return "Sure.. Here are the requested flight airlines"
    elif intent == "atis_airport" and dac == "question": #5 atis_airport
        return "The required airports for "+source+" to "+destination+" are.."
    elif intent == "atis_airport" and dac == "command":
        return "Please find below the requested airports for "+source+" to "+destination
    elif intent == "atis_airport" and dac == "statement":
        return "Sure.. Here are the requested airports for "+source+" to "+destination
    elif intent == "atis_distance" and dac == "question": #6 atis_distance
        return "The required flight durations for "+source+" to "+destination+" are.."
    elif intent == "atis_distance" and dac == "command":
        return "Please find below the flight durations for "+source+" to "+destination
    elif intent == "atis_distance" and dac == "statement":
        return "Sure.. Here are the flight durations for "+source+" to "+destination
    if intent == "atis_aircraft" and dac == "question": #7 atis_aircraft
        return "The required flight air crafts are.."
    elif intent == "atis_aircraft" and dac == "command":
        return "Please find below the requested flight air crafts"
    elif intent == "atis_aircraft" and dac == "statement":
        return "Sure.. Here are the requested flight air crafts"
    if intent == "atis_city" and dac == "question": #8 atis_city
        return "The required flights from `city` are.."
    elif intent == "atis_city" and dac == "command":
        return "Please find below the requested flight from `city`"
    elif intent == "atis_city" and dac == "statement":
        return "Sure.. Here are the requested flights from 'city'"
    if intent == "atis_flight_no" and dac == "question": #9 atis_flight_no
        return "The required flight numbers for "+source+" to "+destination+" are.."
    elif intent == "atis_flight_no" and dac == "command":
        return "Please find below the requested flight numbers for "+source+" to "+destination
    elif intent == "atis_flight_no" and dac == "statement":
        return "Sure.. Here are the requested flight numbers for "+source+" to "+destination
    if intent == "atis_abbreviation" and dac == "question": #10 atis_abbreviation
        return "The required abbreviations are.."
    elif intent == "atis_abbreviation" and dac == "command":
        return "Please find below the requested abbreviations"
    elif intent == "atis_abbreviation" and dac == "statement":
        return "Sure.. Here are the requested abbreviations"
    if intent == "atis_capacity" and dac == "question": #11 atis_capacity
        return "The required flight capacities are.."
    elif intent == "atis_capacity" and dac == "command":
        return "Please find below the requested flight capacities"
    elif intent == "atis_capacity" and dac == "statement":
        return "Sure.. Here are the requested flight capacities"
    if intent == "atis_ground_fare" and dac == "question": #12 atis_ground_fare
        return "The required ground fares in 'city' are.."
    elif intent == "atis_ground_fare" and dac == "command":
        return "Please find below the requested ground fares in 'city'"
    elif intent == "atis_ground_fare" and dac == "statement":
        return "Sure.. Here are the requested ground fares in 'city'"
    if intent == "atis_ground_service" and dac == "question": #13 atis_ground_service
        return "The required ground services in 'city' are.."
    elif intent == "atis_ground_service" and dac == "command":
        return "Please find below the requested ground services in 'city'"
    elif intent == "atis_ground_service" and dac == "statement":
        return "Sure.. Here are the requested ground services in 'city'"
    if intent == "atis_meal" and dac == "question": #14 atis_meal
        return "The required meals on the flights from "+source+" to "+destination+" are.."
    elif intent == "atis_meal" and dac == "command":
        return "Please find below the meals on the flights from "+source+" to "+destination
    elif intent == "atis_meal" and dac == "statement":
        return "Sure.. Here are the requested meals on the flights from "+source+" to "+destination
    if intent == "atis_quantity" and dac == "question": #15 atis_quantity
        return "The total available flights from "+source+" to "+destination+" are.."
    elif intent == "atis_quantity" and dac == "command":
        return "Please find below the total available flights from "+source+" to "+destination
    elif intent == "atis_quantity" and dac == "statement":
        return "Sure.. Here are the total available flights from "+source+" to "+destination
    if intent == "atis_restriction" and dac == "question": #16 atis_restriction
        return "The required restrictions on the flights are"
    elif intent == "atis_restriction" and dac == "command":
        return "Please find below the restrictions on the flights"
    elif intent == "atis_restriction" and dac == "statement":
        return "Sure.. Here are the requested restrictions on the flight"

def applyall(i):
    return np.asarray(i)

def li_to_dict(tup):
    di={}
    for a, b in tup:
        #a.replace("'","")
        #b.replace("'","")
        di.setdefault(b, []).append(a)
    return di
  
def create_intelligent_word_embedding(text):
  charcnn_embedding = []
  bert_embedding = []
  final_embedding = []
  word_list = text.split(" ")

  for i,word in enumerate(word_list):
    bert_embedding.append(bert_model.encode(word))
    charcnn_embedding.append(np.squeeze(charCNN_model.predict(np.expand_dims(charCNN_input_preprocess(word)[0],axis = 0)),axis=0))
    final_embedding.append(np.concatenate([bert_embedding[i],charcnn_embedding[i]]))

  return final_embedding, bert_embedding, charcnn_embedding

model_atis = load_model("./data/atis_multi_task")
charCNN_model = load_model("./data/charCNN")
bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

dac_la=joblib.load('./data/dac_label.joblib')
intent_la=joblib.load('./data/intent_label.joblib')
with open("./data/slot_label", "rb") as fp:
    label_list = pickle.load(fp)
with open("./data/slot_index", "rb") as fp:
    index_list = pickle.load(fp)


@app.route("/")
def index():
    return flask.render_template('index1.html')


@app.route('/predict', methods=['POST'])
def predict():
  app.title = "ATIS Predictions"
  q = request.form.get("input")
  print("Input Query: " + q)
  tokens = list(q.split(" "))
  query_df = pd.DataFrame(columns = ['sen'])
  query_df.loc[0] = [q]

  query_col = []
  a = [query_col.append(text.split(" ")) for text in query_df["sen"]]
  padded = pad_sequences(sequences=query_col, maxlen=24, dtype=object, padding='post', truncating='post', value="<UNK>")
  res = [' '.join(ele) for ele in padded]
  query_df["pad_sen"] = res

  series = query_df["pad_sen"].progress_apply(create_intelligent_word_embedding)
  query_df["intelligent_concatenated_representation"] = pd.DataFrame(series.tolist())[0].tolist()
  query_df["bert_embeddings"] = pd.DataFrame(series.tolist())[1].tolist()
  query_df["charcnn_embeddings"] = pd.DataFrame(series.tolist())[2].tolist()

  X_test = query_df['intelligent_concatenated_representation']
  X_test = X_test.values
  X_test = applyall(X_test)
  X_test = np.stack(X_test)

  prediction = model_atis.predict(np.expand_dims(X_test[0],axis=0))
  dac_prediction = (prediction[0] > 0.5).astype("int32")
  intent_prediction = (prediction[1] > 0.5).astype("int32")
  slots = [label_list[index_list.index(j)] for j in [np.argmax(x) for x in prediction[2][0][:]] if j in index_list]
  dp = str(*dac_la.inverse_transform([np.argmax(dac_prediction, axis=None, out=None)]))
  ip = str(*intent_la.inverse_transform([np.argmax(intent_prediction, axis=None, out=None)]))
  
  pos_tags = [pos for token, pos in pos_tag(tokens)]
  conlltags = [(token, pos, tg) for token, pos, tg in zip(tokens, pos_tags,slots)]
  ne_tree = conlltags2tree(conlltags)

  text = []
  for subtree in ne_tree:
    if type(subtree) == Tree:
        original_label = subtree.label()
        original_string = " ".join([token for token, pos in subtree.leaves()])
        text.append((original_string, original_label))
  
  si = li_to_dict(text)
  sp = str(text)[1:-1] 
  
  b = False
  if 'fromloc.city_name' in si.keys():
    b = True
  else:
    b = False
  if 'toloc.city_name' in si.keys():
    b = True
  else:
    b = False
  
  date=""
  day_number=""
  month= ""
  if 'depart_date.day_name' in si.keys():
    date = str(*si.get("depart_date.day_name"))

    if 'depart_date.day_number' in si.keys():
        day_number = str(*si.get("depart_date.day_number"))
    
    if 'depart_date.month_name' in si.keys():
      month = str(*si.get("depart_date.month_name"))

  print(date)

  error="all locations not specified"
  print(si)
  # if(b == False):
  #   return redirect(url_for('index',error=error))
  heading = ""
  list_p =[dp,ip,sp]
  air_data = [{"airline":"Error all locations not specified"}]
  air_data_round = [{"airline":"Error all locations not specified"}]

  heading = ""

  trip="one way"

  if 'round_trip' in si.keys():
    trip = str(*si.get("round_trip"))
  
  cheap_var="none"
  if 'cost_relative' in si.keys():
    cheap_var = str(*si.get("cost_relative"))

  source = ""
  destination =""
  dep_date = ""
  ret_date = ""
  ret_date_value = ""
  ret_date_round = ""
  if(b == True):
    air_data=[]
    li_img=[]
    img_list=[]

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    driver = webdriver.Chrome(ChromeDriverManager().install(),options=chrome_options)
    # driver = webdriver.Chrome('C:\\Users\\gilbe\\.wdm\\drivers\\chromedriver\\win32\\100.0.4896.60\\chromedriver.exe',options=chrome_options)
    source =str(*si.get("fromloc.city_name"))
    destination =str(*si.get("toloc.city_name"))
    heading = mapping_func(ip,dp,source,destination)
     # with space
    url = "https://www.google.com/travel/flights?q=Flights%20to%20"+destination+"%20from%20"+source+"%20on%20"+date+"%20"+month+"%20"+day_number+"%20"+trip+"%20"
    driver.get(url)
    time.sleep(3)
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    driver.close()

    ret_date_ele = soup.find('input', attrs = {'class' : 'RKk3r eoY5cb j0Ppje','placeholder' : 'Return'})
    ret_date_value = ret_date_ele.get('value')

    all_divs = soup.find('div', attrs = {'class' : 'S0QcGc'})
    img = all_divs.find_all('img', attrs={'class' : 'EbY4Pc'})

    all_divs_dep_date = soup.find_all('input', attrs = {'class' : 'RKk3r eoY5cb j0Ppje','placeholder' : 'Departure'})
    all_divs_ret_date = soup.find_all('input', attrs = {'class' : 'RKk3r eoY5cb j0Ppje','placeholder' : 'Return'})

    for item in all_divs_dep_date:
        dep_date = item.get('value')
        break
    for item in all_divs_ret_date:
        ret_date = item.get('value')
        break
    
    for im in img:
        li_img.append(im['src'])

    li_in = 5
    img_list=[]
    for li in li_img:
        if li_in %5 == 0:
            img_list.append(li)
        li_in = li_in + 1

    listItems = all_divs.find_all('div', attrs = {'role' : 'listitem'})
    airlineNames = all_divs.find_all("span", class_="h1fkLb")

    i = 1
    for item,airline,img_item in zip(listItems,airlineNames,img_list):
        temp_dict={}
        temp_dict["no"] = i
        temp_dict["img"] = img_item
        i = i+1
        # temp_dict["ret_date"] = ret_date

        s = item.find("div", attrs = {'class' : 'U3gSDe'}).text
        if trip == "round trip":
            if(s == "Price unavailable"):
                temp_dict["price"] = "Price Unavailable"
                temp_dict["trip"] = "Trip Unavailable"
            else :
                l = s.split('₹')
                li = "₹"+l[1]
                li=re.split('([a-z]+)',li,maxsplit=1)
                t = li[1]+li[2]
                temp_dict["price"] = li[0]
                temp_dict["trip"] = "round trip"
        if trip == "one way":
            if(s == "Price unavailable"):
                temp_dict["price"] = "Price Unavailable"
                temp_dict["trip"] = "Trip Unavailable"
            else :
                l = s.split('₹')
                if len(l) > 1:
                    li = "₹"+l[1]
                    # li=re.split('([a-z]+)',li,maxsplit=1)
                    # t = li[1]+li[2]
                    temp_dict["price"] = li
                    temp_dict["trip"] = "one way"

        x = item.find("div", attrs = {'class' : 'Ak5kof'})
        temp_dict["duration"] = x.find("div", attrs = {'class' :  ['gvkrdb' ,'AdWm1c', 'tPgKwe', 'ogfYpf'], 'aria-label' : re.compile(r'Total duration *')}).text
        x = item.find("div", attrs = {'aria-label' : re.compile(r'Departure time*')})
        temp_dict["departureTime"] = x.text
        x = item.find("div", attrs = {'aria-label' : re.compile(r'Arrival time*')})
        temp_dict["arrivalTime"] = x.text
        temp_dict["airline"] = airline.text
        x = item.find("div", attrs = {'class' :'G2WY5c sSHqwe ogfYpf tPgKwe'})
        temp_dict["fromCode"] = x.text
        x = item.find("div", attrs = {'class' :'c8rWCd sSHqwe ogfYpf tPgKwe'})
        temp_dict["toCode"] = x.text
        x = item.find("span", attrs = {'class' :'VG3hNb'})
        temp_dict["stoppings"] = x.text
        x = item.find_all("span", attrs = {'class' :'eoY5cb'})
        for stuff,n in zip(x,range(4)):
            if(n == 0):
                s = stuff.text
                s = ' '.join(s.split())
                temp_dict["fromTimings"] = s
            if(n == 1):
                s = stuff.text
                s = ' '.join(s.split())
                temp_dict["toTimings"] = s
            if(n == 2):
                temp_dict["fromAirport"] = stuff.text
            if(n == 3):
                temp_dict["toAirport"] = stuff.text
        air_data.append(temp_dict.copy())
    # driver.close()
    
    print(air_data)     
    cheapest_flight = []
    cheapest_flight.append(air_data[0])


    print("____________________________________")
    print(air_data[0])



  if( trip == 'round trip'):
    air_data_round=[]
    li_img_round=[]
    img_list_round=[]

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    driver = webdriver.Chrome(ChromeDriverManager().install(),options=chrome_options)
    # driver = webdriver.Chrome('C:\\Users\\gilbe\\.wdm\\drivers\\chromedriver\\win32\\100.0.4896.60\\chromedriver.exe',options=chrome_options)
    source =str(*si.get("fromloc.city_name"))
    destination =str(*si.get("toloc.city_name"))
    heading = mapping_func(ip,dp,source,destination)
     # with space
    url = "https://www.google.com/travel/flights?q=Flights%20to%20"+source+"%20from%20"+destination+"%20on%20"+ret_date_value+"%20"
    driver.get(url)
    time.sleep(3)
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    driver.close()

    ret_date_ele = soup.find('input', attrs = {'class' : 'RKk3r eoY5cb j0Ppje'})
    ret_date_value = ret_date_ele.get('value')

    all_divs = soup.find('div', attrs = {'class' : 'S0QcGc'})
    img = all_divs.find_all('img', attrs={'class' : 'EbY4Pc'})

    all_divs_dep_date = soup.find_all('input', attrs = {'class' : 'RKk3r eoY5cb j0Ppje','placeholder' : 'Departure'})
    all_divs_ret_date = soup.find_all('input', attrs = {'class' : 'RKk3r eoY5cb j0Ppje','placeholder' : 'Return'})

    # for item in all_divs_dep_date:
    #     dep_date = item.get('value')
    #     break
    # for item in all_divs_ret_date:
    #     ret_date = item.get('value')
    #     break
    
    for im in img:
        li_img_round.append(im['src'])

    li_in = 5
    img_list_round=[]
    for li in li_img_round:
        if li_in %5 == 0:
            img_list_round.append(li)
        li_in = li_in + 1

    listItems = all_divs.find_all('div', attrs = {'role' : 'listitem'})
    airlineNames = all_divs.find_all("span", class_="h1fkLb")

    i = 1
    for item,airline,img_item in zip(listItems,airlineNames,img_list_round):
        temp_dict={}
        temp_dict["no"] = i
        temp_dict["img"] = img_item
        i = i+1
        # temp_dict["ret_date"] = ret_date

        s = item.find("div", attrs = {'class' : 'U3gSDe'}).text
        if trip == "round trip":
            if(s == "Price unavailable"):
                temp_dict["price"] = "Price Unavailable"
                temp_dict["trip"] = "Trip Unavailable"
            else :
                l = s.split('₹')
                li = "₹"+l[1]
                li=re.split('([a-z]+)',li,maxsplit=1)
                t = li[1]+li[2]
                temp_dict["price"] = li[0]
                temp_dict["trip"] = "round trip"
        if trip == "one way":
            if(s == "Price unavailable"):
                temp_dict["price"] = "Price Unavailable"
                temp_dict["trip"] = "Trip Unavailable"
            else :
                l = s.split('₹')
                if len(l) > 1:
                    li = "₹"+l[1]
                    # li=re.split('([a-z]+)',li,maxsplit=1)
                    # t = li[1]+li[2]
                    temp_dict["price"] = li
                    temp_dict["trip"] = "one way"

        x = item.find("div", attrs = {'class' : 'Ak5kof'})
        temp_dict["duration"] = x.find("div", attrs = {'class' :  ['gvkrdb' ,'AdWm1c', 'tPgKwe', 'ogfYpf'], 'aria-label' : re.compile(r'Total duration *')}).text
        x = item.find("div", attrs = {'aria-label' : re.compile(r'Departure time*')})
        temp_dict["departureTime"] = x.text
        x = item.find("div", attrs = {'aria-label' : re.compile(r'Arrival time*')})
        temp_dict["arrivalTime"] = x.text
        temp_dict["airline"] = airline.text
        x = item.find("div", attrs = {'class' :'G2WY5c sSHqwe ogfYpf tPgKwe'})
        temp_dict["fromCode"] = x.text
        x = item.find("div", attrs = {'class' :'c8rWCd sSHqwe ogfYpf tPgKwe'})
        temp_dict["toCode"] = x.text
        x = item.find("span", attrs = {'class' :'VG3hNb'})
        temp_dict["stoppings"] = x.text
        x = item.find_all("span", attrs = {'class' :'eoY5cb'})
        for stuff,n in zip(x,range(4)):
            if(n == 0):
                s = stuff.text
                s = ' '.join(s.split())
                temp_dict["fromTimings"] = s
            if(n == 1):
                s = stuff.text
                s = ' '.join(s.split())
                temp_dict["toTimings"] = s
            if(n == 2):
                temp_dict["fromAirport"] = stuff.text
            if(n == 3):
                temp_dict["toAirport"] = stuff.text
        air_data_round.append(temp_dict.copy())
    # driver.close()
    
    print(air_data_round)     
    cheapest_flight_round = []
    cheapest_flight_round.append(air_data_round[0])


    print("Return____________________________________")
    print(air_data_round[0])
    print("____________________________________")
    print(url)
    

  if(cheap_var == "cheapest" and trip =="one way"):
    return flask.render_template('predict.html',heading = heading,q=q,dp = dp, ip = ip, sp = sp,list_p = list_p,air_data=cheapest_flight,trip = trip, source = source, destination = destination,dep_date = dep_date,ret_date = ret_date)
  if(cheap_var == "none" and trip =="one way"):
    return flask.render_template('predict.html',heading = heading,q=q,dp = dp, ip = ip, sp = sp,list_p = list_p,air_data=air_data,trip = trip, source = source, destination = destination,dep_date = dep_date,ret_date = ret_date)
  if(cheap_var == "cheapest" and trip =="round trip"):
    return flask.render_template('predict.html',heading = heading,q=q,dp = dp, ip = ip, sp = sp,list_p = list_p,air_data=cheapest_flight,air_data_round=cheapest_flight_round,trip = trip, source = source, destination = destination,dep_date = dep_date,ret_date = ret_date)
  if(cheap_var == "none" and trip =="round trip"):
    return flask.render_template('predict.html',heading = heading,q=q,dp = dp, ip = ip, sp = sp,list_p = list_p,air_data=air_data,air_data_round=air_data_round,trip = trip, source = source, destination = destination,dep_date = dep_date,ret_date = ret_date)
  # return flask.render_template('predict.html',heading = heading,q=q,dp = dp, ip = ip, sp = sp,list_p = list_p,air_data=air_data,trip = trip, source = source, destination = destination,dep_date = dep_date,ret_date = ret_date)
  # return flask.render_template('predict.html',q=q,dp = dp, ip = ip, sp = sp,heading = heading,trip = trip)


def create_app():
  app = Flask(__name__)
  app.run(debug=True, threaded = True)
  bootstrap = Bootstrap(app)

  return app