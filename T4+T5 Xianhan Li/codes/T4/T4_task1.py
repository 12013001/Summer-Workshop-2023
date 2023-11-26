import cv2 as cv
import numpy as np

# load the model
net = cv.dnn.readNetFromCaffe('./data-tutorial 4/bvlc_googlenet.prototxt', './data-tutorial 4/bvlc_googlenet.caffemodel')

# load labels
labels = "./data-tutorial 4/classification_classes_ILSVRC2012.txt"
with open(labels) as f:
    classes = f.readlines()

# load image
image = cv.imread('2.jpg')

# remove the alpha channel
if image.shape[2] == 4:
    image = image[:, :, :3]

# process the image
blob = cv.dnn.blobFromImage(image, 1.0, (224, 224), (104.0, 117.0, 123.0), True, False)

net.setInput(blob)
pred = net.forward()

# get predictions
prediction = np.argmax(pred)
probability = pred[0][prediction]

print("Best prediction:", f'{classes[prediction].strip()}')
print("probability:", f'{probability:.6f}')