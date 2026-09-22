import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from PIL import Image


im= Image.open(r"imagepath")

X= np.array(im)
X0=X[:,:,0]
X1=X[:,:,1]
X2=X[:,:,2]

def comp_fft(rgb, p):  # FFT applied to each colour channel 
    Xh=np.fft.fft2(rgb)
    Xhflat=Xh.flatten()
    ind=np.argsort((np.abs(Xhflat)))[::-1]
    N=int((i*len(Xhflat))/100)
    Xn=np.zeros_like(Xhflat)
    Xn[ind[:N]]=Xhflat[ind[:N]]
    Xnew=Xn.reshape(Xh.shape)
    Xi=np.real(np.fft.ifft2(Xnew))

    return Xi

p=[0.1, 0.5, 1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]   # Percentage of coefficients retained

er=np.array([])   # MSE for each percentage

for i in p:   
    X0i=comp_fft(X0,i)
    X1i=comp_fft(X1,i)
    X2i=comp_fft(X2,i)
    Xs=np.stack((X0i,X1i,X2i), axis=2)
    Xsc=np.clip(Xs, 0, 255).astype(np.uint8)        
    MSE=np.mean((X-Xs)**2)
    er=np.append(er,MSE)


fig=plt.figure()
a1=fig.add_subplot(1,2,1)
a1.imshow(Xsc)

a2=fig.add_subplot(1,2,2)
a2.scatter(p,er, marker='.')

plt.xlabel("Coefficients retained")
plt.ylabel("MSE")
plt.title("FFT Compression")
plt.grid()
plt.show()


