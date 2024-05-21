import numpy as np
import tensorflow as tf

#Pripravimo primere za treniraje in testiranje
mnist = tf.keras.datasets.mnist
#(primeri za trening), (primeri za testiranje)
(img_train, label_train), (img_test, lbl_test) = mnist.load_data()

#Normaliziramo vhodne vektorje, na začetku so elementi števila med 0-250
img_train = tf.keras.utils.normalize(img_train, axis=1)
img_test = tf.keras.utils.normalize(img_test, axis=1)

images = [] #To je dejanski list vhodnih vektorjev za treniranje
images_test = [] #Ta je za testiranje
label = []
label_test = []

#Podatke, ki smo dobili iz mnist data baze popravimo, da so bolj primerni za racunanje
for img, lbl in zip(img_train, label_train):
    l = np.concatenate(img, axis=None) #pretvori vektor iz 28*28 v 784*1
    l.shape = (784, 1) #popravi dimenzije
    images.append(l)

    vector = np.zeros((10,1)) #Vektoriziramo število primera
    vector[lbl] = lbl
    label.append(vector)

#Enako še za testne primere
for img, lbl in zip(img_test, lbl_test):
    l = np.concatenate(img, axis=None) 
    l.shape = (784, 1)
    images_test.append(l)
    
    vector = np.zeros((10,1))
    vector[lbl] = lbl
    label.append(vector)

data = (images, label), (images_test, label_test)