import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.draw import polygon

##1.Smooth gradient image##

x=np.linspace(0,1,256)
sgrad=np.tile(x,(256,1))
plt.imshow(sgrad,cmap='grey')
Image.fromarray((sgrad * 255).astype(np.uint8)).save("sgrad.png")

##2.Sharp edge##

xb=np.zeros((256,128))
xw=np.ones((256,128))
shed=np.concatenate((xb,xw), axis=1)
plt.imshow(shed,cmap='gray')
Image.fromarray((shed * 255).astype(np.uint8)).save("shed.png")


##3.Sine wave##

xs=np.linspace(0,1,256)
f=2
s=np.sin(2*f*xs*(np.pi))
sin=np.tile(s,(256,1))
plt.imshow(sin,cmap='grey')
Image.fromarray(((sin + 1) / 2 * 255).astype(np.uint8)).save("sine2.png")

##4.Stripes##
n=4
xb=np.zeros((256,2**n))
xw=np.ones((256,2**n))
xa=np.concatenate((xb,xw),axis=1)
stripes=np.tile(xa,int(256/(2**(n+1))))
plt.imshow(stripes,cmap='grey')
Image.fromarray((stripes * 255).astype(np.uint8)).save("stripes.png")
##5. Checkerboard##
n=5
v=np.arange(0,256,2**n)
print(v)
vt=np.array([])
for i in v:
    if int((i/2**n)%(2))==0:
        q=np.zeros((2**n))
        
    elif int((i/2**n)%(2))==1:
        q=np.ones((2**n))
    vt=np.concatenate((vt,q),axis=0)

va=np.array([])
for i in v:
    if int((i/2**n)%(2))==1:
        q=np.zeros((2**n))
        
    elif int((i/2**n)%(2))==0:
        q=np.ones((2**n))
    va=np.concatenate((va,q),axis=0)
vv=np.stack((vt,va),axis=0)

vv=np.repeat(vv,2**n,axis=0 )
check=np.tile(vv, (int(256/(2**(n+1))),1))
plt.imshow(check,cmap='grey')
Image.fromarray((check * 255).astype(np.uint8)).save("check.png")

##Random Noise##

rand=np.random.rand(256,256)
plt.imshow(rand,cmap='grey')
Image.fromarray((rand * 255).astype(np.uint8)).save("noise.png")



##Circle##
import numpy as np
import matplotlib.pyplot as plt

n = 256

y, x = np.ogrid[:n, :n]

r = 60

circle = ((x - 128)**2 + (y - 128)**2 <= r**2).astype(float)
print(np.max(circle))
Image.fromarray((circle * 255).astype(np.uint8)).save("circle.png")

##Corner Square##

import numpy as np
import matplotlib.pyplot as plt

x = np.zeros((256, 256))

x[0:64, 0:64] = 1

plt.imshow(x, cmap="gray")
plt.axis("off")
Image.fromarray((x * 255).astype(np.uint8)).save("cornersquare.png")


##Triangle##


x = np.ones((256, 256))

r = [50, 150, 150]
c = [128, 80, 176]

rr, cc = polygon(r, c, shape=x.shape)

x[rr, cc] = 0

plt.imshow(x, cmap="gray")
plt.axis("off")
Image.fromarray((x* 255).astype(np.uint8)).save("triangle.png")
plt.show()
