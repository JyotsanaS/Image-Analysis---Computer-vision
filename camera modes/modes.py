import cv2
import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

dark_list=[]
label=[]
def feature(file):
        img = cv2.imread(file,1)
        dim=img.shape

        #Calculating Skin %age
        lower = np.array([0, 48, 80], dtype = "uint8")
        upper = np.array([20, 255, 255], dtype = "uint8")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5,5),np.uint8)
        skinMask = cv2.erode(skinMask, kernel, iterations = 1)
        skinMask = cv2.dilate(skinMask, kernel, iterations = 1)
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        skin=0
        for g in range(0,skinMask.shape[0]):
            for d in range(0,skinMask.shape[1]):
                if skinMask[g,d]==255:
                    skin=skin+1

        skin=skin*100.0/(skinMask.shape[0]*skinMask.shape[1])
        
        #Calculating Darkness Percentage
        img = cv2.imread(file,0)
        dark=0
        for i in range(0,dim[0]):
            for j in range (0,dim[1]):
                if img[i,j]<50:
                    dark=dark+1

        dark_per=(dark*100.0)/(dim[0]*dim[1])

        #Calculating amount of blur
        roi_left=img[0:dim[0],0:dim[1]/3]
        roi_right=img[0:dim[0],2*dim[1]/3:dim[1]]
        bl=cv2.Laplacian(roi_left, cv2.CV_64F).var()
        br=cv2.Laplacian(roi_right, cv2.CV_64F).var()
        if br<bl:
                bl=br

        
        return [round(skin,2),round(dark_per,2),round(bl,4)]

for k in range(1,20):
    dark_per=feature("n"+str(k)+".jpg")
    dark_list.append(dark_per)
    dark_per=feature("l"+str(k)+".jpg")
    dark_list.append(dark_per)
    dark_per=feature("p"+str(k)+".jpg")
    dark_list.append(dark_per)
    label.append("night")
    label.append("landscape")
    label.append("portrait")
print dark_list
print label
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(dark_list, label)
#Save model
filename = 'finalized_model.sav'
pickle.dump(neigh, open(filename, 'wb'))

filename = 'finalized_model.sav'
neigh = pickle.load(open(filename, 'rb'))
#pyaare photos
predict_img="n21.jpg"
print(neigh.predict(feature(predict_img)))
predict_img="l21.jpg"
print(neigh.predict(feature(predict_img)))
predict_img="p21.jpg"
print(neigh.predict(feature(predict_img)))
#dusht photos
predict_img="n22.jpg"
print(neigh.predict(feature(predict_img)))
predict_img="l22.jpg"
print(neigh.predict(feature(predict_img)))
predict_img="p22.jpg"
print(neigh.predict(feature(predict_img)))
#extra photos
predict_img="l23.jpg"
print(neigh.predict(feature(predict_img)))
predict_img="l24.jpg"
print(neigh.predict(feature(predict_img)))
predict_img="p30.jpg"
print(neigh.predict(feature(predict_img)))
predict_img="p24.jpg"
print(neigh.predict(feature(predict_img)))
