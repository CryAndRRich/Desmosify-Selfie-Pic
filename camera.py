import cv2

def get_selfie(filepath):
    cam = cv2.VideoCapture(0)
    while True:
        _, frame = cam.read()
        cv2.imshow("Webcam", frame)

        k = cv2.waitKey(1)

        #Press space to take a picture
        if k % 256 == 32:
            cv2.imwrite(filepath, frame)
            break

    cam.release()
    cv2.destroyAllWindows()  
