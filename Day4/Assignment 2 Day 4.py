# Import libraries
import cv2
import mediapipe as mp

#Drawing utility
mp_drawing = mp.solutions.drawing_utils
# Hand recognition utility
mp_hands   = mp.solutions.hands
# Using the library Cv2, to open the camera
cap = cv2.VideoCapture(0)

#Running a loop to perform certain conditions if the camera opens up and also print that, the camera is not accessible, if it doesn't open

with mp_hands.Hands( min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
	while cap.isOpened():
		flag, image = cap.read()
		if not flag:
			print('something is wrong with camera')
			break

        # For recognition of hands even when it is flipped
		image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
		
        # Processing the hand image in camera for True and false values 
		image.flags.writeable = False
		results = hands.process(image)
		
		results = hands.process(image)

		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # For recognition of multiple hands by the mode

		if results.multi_hand_landmarks:
			for hand_landmarks in results.multi_hand_landmarks:
				mp_drawing.draw_landmarks( image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


		cv2.imshow('live', image)

        # Waitkey gives an output image in mseconds, but makes the mind appear like a video since it is too fast
		if cv2.waitKey(10) & 0xff == ord('q'):
			break

# Release the camera and destroy all windows    
cap.release()
cv2.destroyAllWindows()
