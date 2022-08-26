import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open("/Users/girishmeghanani/PycharmProjects/ Speech_Emotion_Recognition_with_librosa/modelForPrediction1.pkl",'rb'))

# creating a function for prediction

def Speech_Emotion_Recognition(input_data):
    feature = extract_feature("input_data", mfcc=True, chroma=True, mel=True)

    feature = feature.reshape(1, -1)

    prediction = loaded_model.predict(feature)
    return print(prediction)



def main():

    # giving a title
    st.title('Speech Emotion Recognition Web App')


    # getting the input data  from the user

    Audio = st.file_uploader("Choose a file")



    # code for Prediction
    analysis = ''

    # creating a button for prediction

    if st.button('Wine Quality Result '):
        analysis = Speech_Emotion_Recognition([Audio])


    st.success(analysis)




if __name__=='__main__':
    main()













