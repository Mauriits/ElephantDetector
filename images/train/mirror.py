import cv2
from os import listdir


dir = "selected_images"

img_names = [img for img in listdir(dir)]    

for img_name in img_names:
    img = cv2.imread(dir + "/" + img_name)
        
    if (img is not None):
        cv2.imshow("elephant", img)
        
        while True:
            key = cv2.waitKey(10)
            if key == ord('n'):
                break
            
            if key == ord('r'):
                flipped = cv2.flip(img, 1)
                cv2.imwrite(dir + "/" + img_name, flipped)
                break                