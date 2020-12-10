!git clone https://github.com/dipspilani/Covid-using-Transfer-Learning-web-app/
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import time
from PIL import Image

st.set_page_config(page_title='COVID Diagnosis using X-Ray samples' , page_icon = ':hospital:' , layout='wide')
st.title("COVID-19 Diagnosis using X-Ray samples via Transfer Learning:syringe: :hospital:")
st.sidebar.title("Menu")
st.markdown("This application is a Streamlit dashboard used "
            "for **COVID-19 diagnosis using Neural Nets**")
st.markdown('Developed by Mayank Dipanshu Prashant Tanmay')

 
st.write("**Tensorflow Version (must be higher than 2.2.0): **  ",tf.__version__)                       
st.write("**Keras Version : **  ",keras.__version__)

st.sidebar.subheader('Select Mode')
mode = st.sidebar.radio('Mode' , ('Normal' , 'Code' , 'Statistics'))
if mode=='Code':
    st.write('Follow the link:')
    st.write('https://drive.google.com/file/d/1ndjYY_GoaZ_IM33FyUGiejhbDXBMkRyM/view?usp=sharing')
    st.balloons()
elif mode=='Normal':
    st.header("Models and their performances")
    algos = pd.DataFrame({"Architecture":["VGG16-based" , "MobileNet-based" , "ResNet-based" , "EfficientNet-based" , "Simple CNN"],
                        "Accuracy %": [95.6,97.4,97,94.35,77] , "Sensitivity":[91,95,93,89,59] ,
                        "Specificity" : [12900/130 , 12900/130 , 100 , 12800/130 , 100]})
    st.table(algos)

    st.sidebar.subheader('Steps to use the tool:')
    st.sidebar.info('1. Move the image to the same working directory as the .py file')
    st.sidebar.info('2. Browse and open the file using File Uploader')
    st.sidebar.info('3. Copy file name of the uploaded image in the text box')
    st.sidebar.info('3. Select Model')

    

    @st.cache(allow_output_mutation=True , hash_funcs={tf.keras.models.load_model: id})
    def load_vgg16():
        model2 = tf.keras.models.load_model('vgg16.h5')
        return model2

    @st.cache(allow_output_mutation=True , hash_funcs={tf.keras.models.Model: id})
    def load_mobilenet():
        model3 = tf.keras.models.load_model('mobilenet.h5')
        return model3

    @st.cache(allow_output_mutation=True , hash_funcs={tf.keras.models.Model: id})
    def load_resnet():
        model4 = tf.keras.models.load_model('resnet.h5')
        return model4

    @st.cache(allow_output_mutation=True , hash_funcs={tf.keras.models.Model: id})
    def load_effnet():
        model5 = tf.keras.models.load_model('effnet.h5')
        return model5

        
    st.sidebar.subheader('Select sample file to make predictions upon')    
    texx = st.sidebar.text_input(label="Enter File Name (as selected in the file uploader)" , max_chars = 30 , value = "")
    if "." in texx:
        try:
            xray1 = image.load_img(texx , target_size=(224,224))
            img_array = image.img_to_array(xray1)
            img_batch = np.expand_dims(img_array, axis=0)
            vgg = tf.keras.applications.vgg16.preprocess_input(img_batch)
            mn = tf.keras.applications.mobilenet.preprocess_input(img_batch)
            rn = tf.keras.applications.resnet50.preprocess_input(img_batch)
            en = tf.keras.applications.efficientnet.preprocess_input(img_batch)
        except:
            st.sidebar.markdown('File does not exist in directory/invalid file type')
    else:
        st.sidebar.markdown('Enter file name with its file type (eg abc.png)')  
        



    st.header('Upload X-Ray sample Here')
    sample = st.file_uploader(label="Select File (.png or .jpg)" , type=['png','jpg'])

    if sample is not None:
        try:
            #ximg = Image.open(sample)
            #pix = image.img_to_array(ximg)
            #pix = image.smart_resize(pix,(224,224))
           
            #pix = np.array([pix])
            #vgg = tf.keras.applications.vgg16.preprocess_input(pix)
            #mn = tf.keras.applications.mobilenet.preprocess_input(pix)
            #rn = tf.keras.applications.resnet50.preprocess_input(pix)
            #en = tf.keras.applications.efficientnet.preprocess_input(pix)


            
            st.success('File Uploaded')
        except:
            st.warning('Please choose a valid file')
        xx = st.checkbox('Display Image')
        if xx:
            try:
                st.image(sample.read(),caption="X-Ray sample" , use_column_width=True)
            except:
                st.warning('Invalid Image')
        
        
    st.sidebar.subheader("Get prediction from model")
    model_select = st.sidebar.selectbox('Select Model', ('None','VGG16', 'MobileNet','ResNet50' , 'EfficientNetB0'))  
    if model_select=='VGG16':
        with st.spinner('Loading VGG16..'):
            vmodel = load_vgg16()
            time.sleep(3)
        try:
            b = vmodel.predict(vgg)
            st.write('COVID-19 Probability% :')
            st.write(b[0][0]*100)
            a = np.argmax(b)
            if a==0:
                st.warning("**VGG16 Prediction:** COVID-19 Diagnosed")
            elif a==1:
                st.success("**VGG16 Prediction:** Sample is COVID negative")
        except:
            st.sidebar.markdown('Please choose a valid file first')
        
    elif model_select=='MobileNet':
        with st.spinner('Loading MobileNet..'):
            mmodel = load_mobilenet()
            time.sleep(3)
        try:
            b = mmodel.predict(mn)
            st.write('COVID-19 Probability% :')
            st.write(b[0][0]*100)
            a = np.argmax(b)
            if a==0:
                st.warning("**MobileNet Prediction:** COVID-19 Diagnosed")
            elif a==1:
                st.success("**MobileNet Prediction:** Sample is COVID negative")    
        except:
            st.sidebar.markdown('Please choose a valid file first')
            
    elif model_select=='ResNet50':
        with st.spinner('Loading ResNet50..'):
            rmodel = load_resnet()
            time.sleep(3)
        try:
            b = rmodel.predict(rn)
            st.write('COVID-19 Probability% :')
            st.write(b[0][0]*100)
            a = np.argmax(b[0])
            if a==0:
                st.warning("**ResNet50 Prediction:** COVID-19 Diagnosed")
            elif a==1:
                st.success("**ResNet50 Prediction:** Sample is COVID negative")
        except:
            st.sidebar.markdown('Please choose a valid file first')

            
    elif model_select=='EfficientNetB0':
        with st.spinner('Loading EfficientNetB0..'):
            emodel = load_effnet()
            time.sleep(3)
        try:
            b = emodel.predict(en)
            st.write('COVID-19 Probability% :')
            st.write(b[0][0]*100)
            a = np.argmax(b)
            if a==0:
                st.warning("**EfficientNet Prediction:** COVID-19 Diagnosed")
            elif a==1:
                st.success("**EfficientNet Prediction:** Sample is COVID negative")        

        except:
            st.sidebar.markdown('Please choose a valid file first')


elif mode=='Statistics':
    @st.cache(allow_output_mutation=True)
    def csv1():
        data = pd.read_csv('PatientInfo.csv')
        return data
    
    st.subheader("Statistics from South Korea's data")
    plot_type = st.selectbox('Select Plot', ('None','COVID-19 Age Distribution', 'COVID-19 Gender Distribution' , 'Hotspots' , 'City-Wise'))           

    if plot_type=='COVID-19 Age Distribution':
        df = pd.DataFrame({"Age":["0s","10s","20s","30s","40s","50s","60s","70s","80s"] , "Patients Distribution":[66,178,899,523,518,667,482,232,170]})
        fig = px.pie(df,names="Age" , values="Patients Distribution", color = "Patients Distribution",title="Age Distribution",width=1000,height=600)
        st.plotly_chart(fig)
    elif plot_type=='COVID-19 Gender Distribution':
        df = pd.DataFrame({"Gender":["Male","Female"] , "Patients%":[182500/(2218+1825),221800/(2218+1825)]})
        fig = px.bar(df,x="Gender" , y="Patients%", color = "Patients%",title="Gender Distribution",width=500)
        st.plotly_chart(fig)
    elif plot_type=='Hotspots':
        hs = pd.DataFrame({"lat":[37.56 , 35.87 , 37.41] , "lon":[126.97 , 128.6 , 127.51]})
        st.map(hs,zoom=4)
    elif plot_type=='City-Wise':
        data = csv1()
        q = data.iloc[:,[4,10]]
        q.dropna()
        fig = px.scatter(q,height=600,y="province",x="confirmed_date",labels={"confirmed_date":"Date of Diagnosis"},title="Continuous and Densely populated points mean bigger outbreak")
        st.plotly_chart(fig)
