import cv2
import os

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 32:
        img_name = "frame1.png"
        image_path = os.path.join('frames', img_name)
        cv2.imwrite(image_path, frame)
        print("{} written!".format(img_name))
        break

cam.release()
cv2.destroyAllWindows()  