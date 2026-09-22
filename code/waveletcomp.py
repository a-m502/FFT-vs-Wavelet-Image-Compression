import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from PIL import Image
import pywt 
im= Image.open(r"C:\Users\Abdusamed\Downloads\cat.jpg")

X= np.array(im)

X0=X[:,:,0]
X1=X[:,:,1]
X2=X[:,:,2]

def comp_fft(rgb, p, w, n):   # Haar wavelet transform is applied to each colour channel
    Xh, b= pywt.coeffs_to_array(pywt.wavedec2(rgb,wavelet=w,level=n))
    Xhflat=Xh.flatten()
    ind=np.argsort((np.abs(Xhflat)))[::-1]
    N=int((p*len(Xhflat))/100)
    Xn0=np.zeros_like(Xhflat)
    Xn0[ind[:N]]=Xhflat[ind[:N]]
    Xnew=Xn0.reshape(Xh.shape)
    Xre=pywt.array_to_coeffs(Xnew, b, output_format='wavedec2')
    Xi=pywt.waverec2(Xre, wavelet=w)
    Xi = Xi[:627, :418]
    return Xi


p=[0.1, 0.5, 1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100] # Percentage of coefficients retained
er=np.array([]) # MSE for each percentage

for i in p: 
    X0i=comp_fft(X0,i, 'db1', 2)
    X1i=comp_fft(X1,i,'db1', 2)
    X2i=comp_fft(X2,i,'db1', 2)
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
plt.title("Wavelet Compression")
plt.grid()
plt.show()


