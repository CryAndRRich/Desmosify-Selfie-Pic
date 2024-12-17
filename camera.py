import cv2
import os

def get_selfie():
    cam = cv2.VideoCapture(0)
    while True:
        _, frame = cam.read()
        cv2.imshow("Webcam", frame)

        k = cv2.waitKey(1)

        #Press space to take a picture
        if k % 256 == 32:
            img_name = 'sample0.png'
            image_path = os.path.join('samples', img_name)
            cv2.imwrite(image_path, frame)
            break

    cam.release()
    cv2.destroyAllWindows()  
