# Libraries

# Standard

import os

# 3rd Party Libraries

from PIL import Image


def main():

    directory = "data/mnist-train-images/"

    for f in os.listdir(directory + "tif/"):

        # Create filenames

        tif_name = directory + "tif/" + f

        txt_name = directory + "txt/" + f[:-4] + ".txt"

        # Get pixels

        digit_image = Image.open(tif_name)

        pixels = []

        for i in range(28):

            for j in range(28):

                pixel = digit_image.getpixel((i, j))

                pixel = 1 - (pixel / 255)

                pixels.append(pixel)

        del digit_image

        # Save to text

        text_file = open(txt_name, "w")

        for x in pixels:

            text_file.write(str(x) + "\n")

        text_file.close()

    directory = "data/mnist-test-images/"

    for f in os.listdir(directory + "tif/"):

        # Create filenames

        tif_name = directory + "tif/" + f

        txt_name = directory + "txt/" + f[:-4] + ".txt"

        # Get pixels

        digit_image = Image.open(tif_name)

        pixels = []

        for i in range(28):

            for j in range(28):

                pixel = digit_image.getpixel((i, j))

                pixel = 1 - (pixel / 255)

                pixels.append(pixel)

        del digit_image

        # Save to text

        text_file = open(txt_name, "w")

        for x in pixels:

            text_file.write(str(x) + "\n")

        text_file.close()


if (__name__ == "__main__"):

    main()
