from fastai.vision.all import *
import streamlit as st
import pathlib
import plotly.express as px
import platform 
plt=platform.system()
if plt=='Linux':pathlib.WindowsPath=pathlib.PosixPath
st.title('Buyumlarni klassifikatsiyalovchi model')
file=st.file_uploader("Rasm yuklash:,type=['jpeg','jpg','svg','gif']")
if file:
    st.image(file)
    img=PILImage.create(file)
    model=load_learner("mustaqil_model.pkl")
    pred,pred_id,probs=model.predict(img)
    st.success(f"Bashorat:{pred}")
    st.info(f"Ehtimollik:{probs[pred_id]*100}%")
    fig=px.bar(x=probs*100,y=model.dls.vocab)
    st.plotly_chart(fig)
