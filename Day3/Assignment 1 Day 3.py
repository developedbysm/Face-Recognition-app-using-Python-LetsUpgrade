import cv2
import mediapipe as mp
import numpy as np

#Mediapipe, a cross-platform framework for building multimodal applied machine learning pipelines - video, audio, time series etc

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Getting the user input to change the background of the live camera, according to the input

print('Enter the background you want \nPress 1 for Background 1 \nPress 2 for Background 2 \nPress 3 for Background 3 \nPress 4 for Background 4 \nPress 5 for Background 5 \nPress 6 for Background 6')

# Get the input from the user

inp = int(input())

# Running a loop to change background according to the input

if inp == 1 :
    bg_img = cv2.imread('Images\Background 1.jpg')

elif inp == 2:
    bg_img = cv2.imread('Images\Background 2.jpg')

elif inp == 3:
    bg_img = cv2.imread('Images\Background 3.jpg')

elif inp == 4:
    bg_img = cv2.imread('Images\Background 4.jpg')

elif inp == 5:
    bg_img = cv2.imread('Images\Background 5.jpg')

elif inp == 6: 
    bg_img = cv2.imread('Images\Background 6.jpg')
    
model = mp_selfie_segmentation.SelfieSegmentation(model_selection = 1)

# Using the library Cv2, to open the camera
cap = cv2.VideoCapture(0)

#Running a loop to perform certain conditions if the camera opens up and also print that, the camera is not accessible, if it doesn't open

while cap.isOpened():
    # read the background
    Flag, Frame = cap.read()
    if not Flag:
        print('Error')
        break

    results = model.process(Frame)

    condition = np.stack((results.segmentation_mask,)*3, axis =- 1) > 0.1
    if bg_img is None:
        print('something went wrong try again')
        break

    # Resize the image to the shape of the frame

    bg_img = cv2.resize(bg_img, (Frame.shape[1], Frame.shape[0]))
    output_img = np.where(condition, Frame, bg_img)
 
    cv2.imshow('Frame', output_img)
    # Press q to break
    if cv2.waitKey(10) & 0xff == ord('q'):
        break
# Release the camera and destroy all windows    
cap.release()
cv2.destroyAllWindows()
