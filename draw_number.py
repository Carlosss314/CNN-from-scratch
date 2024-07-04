import numpy as np
import matplotlib.pyplot as plt

from tkinter import *
import PIL
from PIL import Image, ImageDraw

def draw_number():
    def paint(event):
        x1, y1 = (event.x + 1), (event.y + 1)
        x2, y2 = (event.x - 1), (event.y - 1)
        cv.create_rectangle(x1, y1, x2, y2, fill="black", width=10)  # On tkinter Canvas
        draw.rectangle((x1/10, y1/10, x2/10, y2/10), fill="white", width=1)  # On PIL Canvas

    def save():
        image1.save("image.png")

    app = Tk()

    cv = Canvas(app, width=280, height=280, bg='white')
    cv.pack()

    image1 = Image.new("L", (28, 28), (0))
    draw = ImageDraw.Draw(image1)

    cv.bind("<B1-Motion>", paint)

    button=Button(text="save", command=save)
    button.pack()

    app.mainloop()

def read_image():
    image = PIL.Image.open("image.png")
    image_array = np.array(image) / 255
    return image_array

def model_parameters(fichier):
    w = np.load(f"{fichier}/w.npy")
    b = np.load(f"{fichier}/b.npy")
    filter = np.load(f"{fichier}/filter.npy")
    return w, b, filter


#functions needed to make predictions
def conv(input_img, filter):
    feature_maps = np.zeros((input_img.shape[0]-filter.shape[0]+1, input_img.shape[0]-filter.shape[0]+1, filter.shape[2]))

    for p in range(filter.shape[2]):
        for i in range(feature_maps.shape[0]):
            for j in range(feature_maps.shape[1]):
                    if (i+3<feature_maps.shape[0] and j+3<feature_maps.shape[1]):
                        patch_of_img = input_img[i:i+3,j:j+3]
                        patch=np.multiply(patch_of_img, filter[:,:,p])
                        feature_maps[i,j,p]=np.sum(patch)

    return feature_maps

def ReLU(feature_maps):
    return np.maximum(feature_maps, 0)

def softmax(z):
    return np.exp(z - max(z))/np.sum(np.exp(z - max(z)))

def forward_prop(img, filter, w, b):
    feature_maps = conv(img, filter)
    feature_maps_relu = ReLU(feature_maps)

    z = np.zeros((w.shape[0], 1))
    for i in range(w.shape[0]):
        z[i] = np.sum(np.multiply(w[i,:,:,:], feature_maps_relu)) + b[i]

    a = softmax(z)
    prediction = np.argmax(a)

    return prediction, a




draw_number()
image_array = read_image()
w, b, filter = model_parameters("weights_and_bias2")

predicted_value, a = forward_prop(image_array, filter, w, b)



plt.figure()
plt.imshow(image_array, cmap="Greys")
plt.title(f"prediction: {predicted_value}     probabilité: {(np.max(a) * 100):.2f}%")
plt.show()
