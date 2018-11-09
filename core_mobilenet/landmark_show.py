import sys
import os
import cv2
f=open("/home-ex/tclhk/guoqiang/Animoji/data/training_data/L48_mobilenet/landmark_animoji_LONet.txt")

for i in range(0,30):
    # lines=f.readlines()
    line=f.readline()
    str=line.strip("\n").split(" ")
    landmark=str[2:142]
    # box=str[196:200]
    path_file = str[0]
    # relative_path,name=path_file.split("/")
    # image_path=os.path.join(r"E:\TCL_code\dataset\WFLW_images",path_file)
    img=cv2.imread(path_file)
    h,w,c=img.shape
    print(h,w,c)
    # cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,250,0))
    for i in range(0,70):
        cv2.circle(img,(int(float(landmark[2*i])*w),int(float(landmark[2*i+1])*h)),1,(0,250,0))
    cv2.imshow(path_file,img)
k=cv2.waitKey(0)
if k&0xff==ord("q"):
    cv2.destroyAllWindows()

