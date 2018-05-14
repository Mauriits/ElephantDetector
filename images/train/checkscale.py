import cv2
from os import listdir


dir = "positive_new"

img_names = [img for img in listdir(dir)]    

for img_name in img_names:
    img = cv2.imread(dir + "/" + img_name)
    
    resized = cv2.resize(img, (48, 40))
    resized2 = cv2.resize(img, (96, 80))
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    blurred2 = cv2.GaussianBlur(resized2, (3, 3), 0)
    
    cv2.imshow("actual", img)
    
    cv2.imshow("show", blurred)
    cv2.imshow("show2", resized)
    
    cv2.imshow("show22", blurred2)
    cv2.imshow("show222", resized2)
    
    while True:
        key = cv2.waitKey(20)
        
        if key == ord('n'):
            break
        
        if key == ord('d'):
            cv2.imwrite("positives_deleted/" + img_name, img)
            break