import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)

img = img.flatten() / 255.0

np.savetxt("test.txt", img, fmt='%.14f')