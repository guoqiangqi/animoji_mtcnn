import sys
import os
import cv2

file_path=os.path.dirname(os.path.abspath(__file__))
# f_in=open(os.path.join(file_path,"training_data/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt"),'r')
# f_in=open(os.path.join(file_path,"training_data/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt"),'r')
# f_out=open(os.path.join(file_path,"training_data/landmarks_70/ImageList_3.txt"),'a')
f_in=open(os.path.join(file_path,"training_data/landmarks_70/ImageList_3.txt"),'r')

lines=f_in.readlines()
for line in lines[8870:-1]:
    line_tmp=line.strip("\n").split(" ")
    # landmark=line_tmp[0:196]
    # box=line_tmp[196:200]
    # path_file = line_tmp[206]
    landmark=line_tmp[5:145]
    box=line_tmp[1:5]
    path_file = line_tmp[0]
    
    # relative_path,name=path_file.split("/")
    image_path=os.path.join(file_path,"training_data/landmarks_70/",path_file)
    img=cv2.imread(image_path)
    cv2.rectangle(img,(int(box[0]),int(box[2])),(int(box[1]),int(box[3])),(0,0,255))
    for i in range(0,70):        
        cv2.putText(img,str(i),(int(float(landmark[2*i])),int(float(landmark[2*i+1]))),cv2.FONT_HERSHEY_TRIPLEX,0.3,(0,0,255))        
        cv2.circle(img,(int(float(landmark[2*i])+0.5),int(float(landmark[2*i+1])+0.5)),1,(0,250,0))
    cv2.imshow(path_file,img)
    k=cv2.waitKey(0)
    if k&0xff==ord("q"):
        cv2.destroyAllWindows()
        exit()
    if k&0xff==ord("n"):
        pass    

    # f_out.write('WFLW_images/'+str(path_file) + ' ')
   # #x1,x2,y1,y2
    # f_out.write(str(box[0]) + ' ')
    # f_out.write(str(box[2]) + ' ')
    # f_out.write(str(box[1]) + ' ')
    # f_out.write(str(box[3]) + ' ')
    # for i in range(17):
        # f_out.write(str(landmark[4*i])+' ')
        # f_out.write(str(landmark[4*i+1])+' ')
    # for i in range(33,38): 
        # f_out.write(str(landmark[2*i])+' ')
        # f_out.write(str(landmark[2*i+1])+' ')
    # for i in range(42,47): 
        # f_out.write(str(landmark[2*i])+' ')
        # f_out.write(str(landmark[2*i+1])+' ')
    # for i in range(51,60): 
        # f_out.write(str(landmark[2*i])+' ')
        # f_out.write(str(landmark[2*i+1])+' ')
    # f_out.write(str(landmark[2*60])+' ')
    # f_out.write(str(landmark[2*60+1])+' ')    
    # for i in range(62,67): 
        # f_out.write(str(landmark[2*i])+' ')
        # f_out.write(str(landmark[2*i+1])+' ')
    # f_out.write(str(landmark[2*68])+' ')    
    # f_out.write(str(landmark[2*68+1])+' ')
    # for i in range(70,75): 
        # f_out.write(str(landmark[2*i])+' ')
        # f_out.write(str(landmark[2*i+1])+' ')   
    # for i in range(76,96): 
        # f_out.write(str(landmark[2*i])+' ')
        # f_out.write(str(landmark[2*i+1])+' ')            
    # f_out.write(str(landmark[2*97])+' ')      
    # f_out.write(str(landmark[2*97+1])+' ')
    # f_out.write(str(landmark[2*96])+' ')      
    # f_out.write(str(landmark[2*96+1])+' ')    
    # f_out.write('\t\n')

f_in.close()
# f_out.close()