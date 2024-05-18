import numpy as np
import tensorflow as tf

#Pripravimo primere za treniraje in testiranje
mnist = tf.keras.datasets.mnist
#(primeri za trening), (primeri za testiranje)
(img_train, label_train), (img_test, label_test) = mnist.load_data()

#Normaliziramo vhodne vektorje, na začetku so elementi števila med 0-250
img_train = tf.keras.utils.normalize(img_train, axis=1)
img_test = tf.keras.utils.normalize(img_test, axis=1)

images = [] #To je dejanski list vhodnih vektorjev za treniranje
images_test = [] #Ta je za testiranje

#Podatke, ki smo dobili iz mnist data baze popravimo, da so bolj primerni za racunanje
for example in img_train:
    l = np.concatenate(example, axis=None) #pretvori vektor iz 28*28 v 784*1
    l.shape = (784, 1) #popravi dimenzije
    images.append(l)

#Enako še za testne primere
for example in img_test:
    l = np.concatenate(example, axis=None) 
    l.shape = (784, 1)
    images_test.append(l)

data = (images, label_train), (images_test, label_test)