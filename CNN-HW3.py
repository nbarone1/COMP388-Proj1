#!/usr/bin/env python
# coding: utf-8

# CNN for Equations

# In[70]:
# import statements

from calendar import EPOCH
from random import shuffle
import pandas as pd
from tabnanny import verbose
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import pandas
import numpy as np
import cv2
from keras.utils.np_utils import to_categorical

i_train=pd.read_csv('/content/train_final.csv',index_col=False)
labels=(i_train['784'])
i_train.drop(i_train.columns[[784]],axis=1,inplace=True)
labels=np.array(labels)
cat=to_categorical(labels,num_classes=13)
Img=[]
for i in range(47504):
    Img.append(np.array(i_train[i:i+1]).reshape(28,28,1))


# In[71]:


# Set batch size, loss model, trainer, initializer (for the ReLU)
batch_size, lr, num_epochs = 128, 0.0002, 10
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
updater = tf.keras.optimizers.Adam(learning_rate=lr)
initializer = tf.keras.initializers.HeNormal()
r = list(range(1,num_epochs+1))


# In[72]:


# Task 1

lenet = tf.keras.models.Sequential([
    # fist Conv
    tf.keras.layers.Conv2D(30, (1, 1), input_shape=(28, 28,1), activation='relu'),
    #first pool
    tf.keras.layers.AvgPool2D(pool_size=(2,2)),
    #second Conv
    tf.keras.layers.Conv2D(15, (1, 1), activation='relu'),
    #second pool
    tf.keras.layers.AvgPool2D(pool_size=(2,2)),
    # Dropout
    tf.keras.layers.Dropout(rate=0.5),
    #flatten and dense
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(60,activation='relu'),
    tf.keras.layers.Dense(13)])


# In[73]:


lenet.compile(optimizer=updater, loss = loss, metrics=['accuracy'])
lenet.fit(np.array(Img),cat,epochs=num_epochs,batch_size=batch_size,shuffle=True,verbose=1)

## breaking the image down falls under the object detection, so we instead chose to break the image down by hand
timg1=cv2.imread('/test.jpeg',cv2.IMREAD_GRAYSCALE)


if timg1 is not None:
    #images.append(img)
    img=~timg1
    ret,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ctrs,ret=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt=sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    w=int(28)
    h=int(28)
    train_data=[]
    #print(len(cnt))
    rects=[]
    for c in cnt :
        x,y,w,h= cv2.boundingRect(c)
        rect=[x,y,w,h]
        rects.append(rect)
    #print(rects)
    bool_rect=[]
    for r in rects:
        l=[]
        for rec in rects:
            flag=0
            if rec!=r:
                if r[0]<(rec[0]+rec[2]+10) and rec[0]<(r[0]+r[2]+10) and r[1]<(rec[1]+rec[3]+10) and rec[1]<(r[1]+r[3]+10):
                    flag=1
                l.append(flag)
            if rec==r:
                l.append(0)
        bool_rect.append(l)
    #print(bool_rect)
    dump_rect=[]
    for i in range(0,len(cnt)):
        for j in range(0,len(cnt)):
            if bool_rect[i][j]==1:
                area1=rects[i][2]*rects[i][3]
                area2=rects[j][2]*rects[j][3]
                if(area1==min(area1,area2)):
                    dump_rect.append(rects[i])
    #print(len(dump_rect)) 
    final_rect=[i for i in rects if i not in dump_rect]
    #print(final_rect)
    for r in final_rect:
        x=r[0]
        y=r[1]
        w=r[2]
        h=r[3]
        im_crop =thresh[y:y+h+10,x:x+w+10]
        

        im_resize = cv2.resize(im_crop,(28,28))

        im_resize=np.reshape(im_resize,(1,28,28))
        train_data.append(im_resize)


## this takes builds or eq based on the result from the image capture
eq=''
for i in range(len(train_data)):
    train_data[i]=np.array(train_data[i])
    train_data[i]=train_data[i].reshape(1,28,28,1)
    result=lenet.predict(train_data[i])
    res = np.argmax(result,axis=1)
    if(res==10):
        eq=eq+'-'
    if(res==11):
        eq=eq+'+'
    if(res==12):
        eq=eq+'*'
    if(res[0]==0):
        eq=eq+'0'
    if(res[0]==1):
        eq=eq+'1'
    if(res[0]==2):
        eq=eq+'2'
    if(res[0]==3):
        eq=eq+'3'
    if(res[0]==4):
        eq=eq+'4'
    if(res[0]==5):
        eq=eq+'5'
    if(res[0]==6):
        eq=eq+'6'
    if(res[0]==7):
        eq=eq+'7'
    if(res[0]==8):
        eq=eq+'8'
    if(res[0]==9):
        eq=eq+'9'
   
print(eq)
eval(eq)