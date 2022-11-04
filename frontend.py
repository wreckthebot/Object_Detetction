
import streamlit as st
from PIL import Image
import cv2 
import numpy as np
from skimage import io, color
import os



def main():

    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Face Detection','Team' )
    )
    
    if selected_box == 'Team':
        welcome() 
    
    if selected_box == 'Face Detection':
        face_detection()
 

 

def welcome():
    
    st.title('Image Processing using Streamlit')
    
    st.subheader('A simple app that shows different image processing algorithms. You can choose the options'
             + ' from the left. I have implemented only a few to show how it works on Streamlit. ' + 
             'You are free to add stuff to this app.')
    
    st.image('hackershrine.jpg',use_column_width=True)


def load_image(filename):
    image = cv2.imread(filename)
    return image
 

 

def face_detection():
    
    st.header("Face Detection using haarcascade")

    img_data = st.file_uploader(label='Drag and Drop Image',type =['png','jpg','jpeg'])
    
    if img_data is not None :
        
        
        uploaded_image = Image.open(img_data)

#save uploaded image locally
        uploaded_image.save("./tempimage/userinput.jpg")
        

#display image
        st.image(uploaded_image,caption = 'Your Image', use_column_width= True)

        image2= io.imread("./tempimage/userinput.jpg")

        
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        #eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
        cat_cascade = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
        faces = face_cascade.detectMultiScale(image2)
        #eyes = eye_cascade.detectMultiScale(image2)
        cats = cat_cascade.detectMultiScale(image2)
        #print(f"{len(faces)} faces detected in the image.")
        for x, y, width, height in faces:
            cv2.rectangle(image2, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)
        #for x, y, width, height in eyes:
            #cv2.rectangle(image2, (x, y), (x + width, y + height), color=(0, 255, 0), thickness=2)
        for x, y, width, height in cats:
            cv2.rectangle(image2, (x, y), (x + width, y + height), color=(0, 0, 255), thickness=2)

        cv2.imwrite("./tempimage/userinput.jpg", image2)
        
    st.image(image2, use_column_width=True,clamp = True)
    os.remove("./tempimage/userinput.jpg")
 



    
    

    
    
    
    
if __name__ == "__main__":
    main()