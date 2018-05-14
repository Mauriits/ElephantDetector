import cv2
from os import listdir
import pickle

ix, iy = 0, 0
update = False

# mouse callback function
def draw_at_mouse(event,x,y,flags,param):
    global ix, iy, update
    if event == cv2.EVENT_LBUTTONDOWN:
        #cv2.circle(img,(x,y),100,(255,0,0),-1)
        update = True
        ix, iy = x, y
        
src_dir = "positive"

img_names = [img for img in listdir(src_dir)]

bboxes = []
q = -1
done = False

for img_name in img_names:
    img = cv2.imread(src_dir + "/" + img_name)
    
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_at_mouse)         
     
    if (img is not None):
        bboxes.append([])
        q += 1
        basecanvas = img.copy()
        cv2.imshow("image", img)
        width = 48
        height = 40            
        
        update = False
        
        while True:
            cv2.imshow("image", basecanvas)
            key = cv2.waitKey(50) & 0xFF
            
            if key == 27:
                break   
                
            if key == ord('e') and width + 6 + ix <= img.shape[1] and height + 5 + iy <= img.shape[0]:
                width += 6
                height += 5
                update = True
            
            if key == ord('q') and width - 6 >= 48 and height - 5 >= 40:
                width -= 6
                height -= 5
                update = True                
            
            if key == ord('p'):
                cv2.rectangle(img, (ix, iy), (ix + width, iy + height), (0, 0, 255), 1)
                
                x2 = ix + width
                y2 = iy + height
                
                bboxes[q].append((ix, iy, x2, y2))
                
            if key == ord('n'):
                break            
            
            if key == ord('y'):
                done = True
                break
                
            if key == ord('z') and len(bboxes[q]) > 0:
                del bboxes[q][-1]
                print(len(bboxes[q]))
            
            if update:
                basecanvas = img.copy()
                cv2.rectangle(basecanvas, (ix, iy), (ix + width, iy + height), (255, 255, 0), 2)
                update = False
    if done:
        break

with open("bounding_boxes", 'wb') as f:
    pickle.dump(bboxes, f)
    
