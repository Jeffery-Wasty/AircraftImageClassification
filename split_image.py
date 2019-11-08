from PIL import Image
import os
import cv2
import keyboard
import matplotlib.pyplot as plt

# subdivide
# subdivide a image into subimages {size} in size
# Inputs:
#   path: the path for the input and outputs
#   fileN: the folder and file name
#   suffix: .JPG or .png
#   size: in pixels (e.g. 180 for 180x180)
#   startCount: for labeling, default is 0


def subdivide(path='/', fileN='', suffix='.png', size=180, startCount=0):
    infile = path + fileN + "/" + fileN + suffix
    # e.g. infile = './splits/to_split/DJI_0950/DJI_0950.JPG'
    outfile = path + fileN + "/" + "cracks"
    # e.g. outfile = './splits/to_split/DJI_0950/cracks.JPG'
    chopsize = size

    img = Image.open(infile)
    width, height = img.size
    count = startCount

    # Save Chops of original image
    for x0 in range(0, width, chopsize):
        for y0 in range(0, height, chopsize):
            box = (x0, y0,
                   x0+chopsize if x0+chopsize < width else width - 1,
                   y0+chopsize if y0+chopsize < height else height - 1)
            print('%s %s' % (infile, box))
            img_sub = img.crop(box)
            img.crop(box).save(outfile + "_" + str(count) + suffix)
            count += 1


def classify_subimage(path='/'):
    X = []  # Subpicture array

    for p in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, p), cv2.IMREAD_GRAYSCALE)
        X.append(img_array)

    for i in range(len(X)):
        plt.imshow(X[i], cmap="gray")
        plt.show()
        keyboard.on_press_key("0", lambda _:
                              X[i].replace("cracks", "cracks"))
        keyboard.on_press_key("1", lambda _:
                              X[i].replace("cracks", "screws"))
        keyboard.on_press_key("2", lambda _:
                              X[i].replace("cracks", "joints"))


# subdivide('./splits/', 'DJI_0125', '.png', 180)
classify_subimage('./splits/DJI_0125', 'DJI_0125', '.png', 180)
