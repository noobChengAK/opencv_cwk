import os
import cv2
import numpy as np
import codecs
from matplotlib import pyplot as plt
def Conv2(img, H, W, kernel, n):
    col = np.zeros(H)
    row = np.zeros(W + 2)
    img = np.insert(img, W, values=col, axis=1)
    img = np.insert(img, 0, values=col, axis=1)
    img = np.insert(img, H, values=row, axis=0)
    img = np.insert(img, 0, values=row, axis=0)
    res = np.zeros([H,W],dtype=np.float32)
    for i in range(H):
        for j in range(W):
            temp = img[i:i + 3, j:j + 3]
            temp = np.multiply(temp,kernel)
            res[i][j] = temp.sum()

    return res

def ComputeGrad(src,flag):
    if flag==0:
        kernel=np.zeros([3,3],dtype=np.float32);
        kernel[0,0]=-1;
        kernel[0,1]=-2;
        kernel[0,2]=-1;
        kernel[2,0]=1;
        kernel[2,1]=2;
        kernel[2,2]=1;
        return Conv2(img,img.shape[0],img.shape[1],kernel,3)
    if flag==1:
        kernel=np.zeros([3,3],dtype=np.float32);
        kernel[0,0]=-1;
        kernel[1,0]=-2;
        kernel[2,0]=-1;
        kernel[0,2]=1;
        kernel[1,2]=2;
        kernel[2,2]=1;
        return Conv2(img,img.shape[0],img.shape[1],kernel,3)

def FindEdge(img):
    gradx=ComputeGrad(img,1);
    grady=ComputeGrad(img,0);
    t=np.square(gradx)+np.square(grady)
    t=np.sqrt(t)
    ret,m=cv2.threshold(t,200,255,cv2.THRESH_BINARY)
    return m
img=cv2.imread('kitty.bmp',cv2.IMREAD_ANYDEPTH)
kernel=np.ones([3,3],dtype=np.float32)/9;
dst=Conv2(img,img.shape[0],img.shape[1],kernel,3);
gradx=ComputeGrad(img,1);
grady=ComputeGrad(img,0);
t=np.square(gradx)+np.square(grady)
t=np.sqrt(t)
ret,m=cv2.threshold(t,200,255,cv2.THRESH_BINARY)
plt.hist(t.ravel(),256,[0,512])
plt.show()
cv2.imwrite("t.tiff",t)
cv2.imshow("show",np.uint8(m))
cv2.waitKey(0)
cv2.imwrite("edge.png",m)

# mean method 

gradx=ComputeGrad(dst,1);
grady=ComputeGrad(dst,0);
t2=np.square(gradx)+np.square(grady)
t2=np.sqrt(t)
ret,m2=cv2.threshold(t2,15,255,cv2.THRESH_BINARY)
plt.hist(t2.ravel(),256,[0,512])
plt.show()
cv2.imshow("show",np.uint8(m2))
cv2.waitKey(0)
cv2.imwrite("edge_smooth.png",m2)


# gaosian method

kernel=np.zeros([3,3],dtype=np.float32);
kernel[0,0]=1;
kernel[2,2]=1;
kernel[0,2]=1;
kernel[2,0]=1;
kernel[0,1]=2;
kernel[1,0]=2;
kernel[2,1]=2;
kernel[1,2]=2;
kernel[1,1]=4;
kernel=kernel/16;
dst2=Conv2(img,img.shape[0],img.shape[1],kernel,3);
gradx=ComputeGrad(dst2,1);
grady=ComputeGrad(dst2,0);
t3=np.square(gradx)+np.square(grady)
t3=np.sqrt(t)
ret,m3=cv2.threshold(t3,15,255,cv2.THRESH_BINARY)
plt.hist(t3.ravel(),256,[0,512])
plt.show()
cv2.imshow("show",np.uint8(m3))
cv2.waitKey(0)
cv2.imwrite("edge_gaussian.png",m3)


#path = 'D:\\images\\'
#dirs = os.listdir(path)
#f = codecs.open("totalList.txt", 'wb', 'utf-8')
#count=0
#for dir in dirs:
   
#    count=0
#    subdirs=os.listdir(path+dir);
#    for sub in subdirs:
#        #os.rename(path+dir+'\\'+sub,path+dir+'\\'+str(count)+'.jpg')
#        f.write(path+dir+'\\'+str(count)+'.jpg'+ os.linesep)
#        count=count+1
##    os.system("move "+dir+" D:\\dstImage\\"+str(count))
    

#f.close()
##for dir in dirs:
##     count=count+1
##     os.rename(path+dir,path+str(count));
