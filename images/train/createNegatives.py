import cv2
from os import listdir

src_dir = "."
dst_dir = "./negative_generated"

img_names = [img for img in listdir(src_dir)]


for img_name in img_names:
    img = cv2.imread(src_dir + "/" + img_name)
    basecanvas = img.copy()
    counter = 0
     
    if (img is not None):
        cv2.imshow("image", img)
        x = 0
        y = 0
        width = 48
        height = 40
            
        hspeed = int(width * 0.2)
        vspeed = int(height * 0.2)
        
        while True:
            key = cv2.waitKey(500)
            
            if key == ord('j'):
                hspeed = int(width * 0.2)
                vspeed = int(height * 0.2)  
            
            if key == ord('g') and vspeed > 0 and hspeed > 0:
                hspeed -= int(width * 0.02)
                vspeed -= int(height * 0.02)
                print(hspeed)
                print(vspeed)
            if key == ord('h') and vspeed < height and hspeed < width:
                hspeed += int(width * 0.02)
                vspeed += int(height * 0.02)
                print(hspeed)
                print(vspeed)
            
            
            if key == ord('a') and x > 0:
                newx = x - hspeed
                x = max(0, newx)
            
            if key == ord('d') and x < img.shape[1] - width:
                newx = x + hspeed
                x = min(img.shape[1] - width, newx)                
                
            if key == ord('w') and y > 0:
                newy = y - vspeed
                y = max(0, newy)
            
            if key == ord('s') and y < img.shape[0] - height:
                newy = y + vspeed
                y = min(img.shape[0] - height, newy)
            
            if key == ord('e') and width + 6 + x <= img.shape[1] and height + 5 + y <= img.shape[0]:
                width += 6
                height += 5     
            
            if key == ord('q') and width - 6 >= 48 and height - 5 >= 40:
                width -= 6
                height -= 5        
            
            if key == ord('p'):
                cv2.rectangle(basecanvas, (x, y), (x + width, y + height), (0, 0, 255), 1)
                
                x2 = x + width
                y2 = y + height
    
                cropped_img = img[y:y2, x:x2]
                cv2.imwrite(dst_dir + "/" + str(counter) + img_name, cropped_img)
                counter += 1
            if key == ord('n'):
                break
            
            canvas = basecanvas.copy()
            cv2.rectangle(canvas, (x, y), (x + width, y + height), (255, 255, 0), 2)
            cv2.imshow("image", canvas)