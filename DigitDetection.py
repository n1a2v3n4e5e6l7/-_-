import cv2 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time

if(not os.environ.get('PYTHONHTTPSVERIFY','') and getattr(ssl,'_create_unverified_context',None)):
    ssl._create_default_https_context = ssl._create_unverified_context

x,y = fetch_openml('mnist_784',version = 1,return_X_y = True)
print(pd.Series(y).value_counts())
classes = ['0','1','2','3','4','5','6','7','8','9']
nclasses = len(classes)

xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state = 9,train_size = 7500,test_size = 2500)
xtrainscaled = xtrain/255
xtestscaled = xtest/255

clf = LogisticRegression(solver = 'saga',multi_class = 'multinomial').fit(xtrainscaled,ytrain)

ypred = clf.predict(xtestscaled)
accuracy = accuracy_score(ytest,ypred)
print("Accuracy; ",accuracy)

cap = cv2.VideoCapture(0)
while(True):
    try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width = gray.shape
        upperleft = (int(width/2-56),int(height/2-56))
        bottomright = (int(width/2+56),int(height/2+56))
        cv2.rectangle(gray,upperleft,bottomright,(0,255,0),2)
        roi = gray[upperleft[1]:bottomright[1],upperleft[0]:bottomright[0]]
        imPIL = Image.fromarray(roi)
        imagebw = imPIL.convert('L')
        imagebwresize = imagebw.resize((28,28),Image.ANTIALIAS)
        imagebwinvert = PIL.ImageOps.invert(imagebwresize)
        pixelfilter = 20
        minpixel = np.percentile(imagebwinvert,pixelfilter)
        imagescaled = np.clip(imagebwinvert-minpixel,0,255)
        maxpixel = np.max(imagebwinvert)
        imagescaled = np.asarray(imagescaled)/maxpixel
        testsample = np.array(imagescaled).reshape(1,784)
        testpred =  clf.predict(testsample)
        print("Predicted Class Is: ",testpred)
        cv2.imshow(frame,gray)
        if cv2.waitkey(1)& 0xFF == ord('q'):
            break
        
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()