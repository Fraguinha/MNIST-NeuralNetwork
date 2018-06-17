# Libraries
# Standard

import json
import os

# 3rd Party Libraries

from PIL import Image


def main():

    directory = "data/mnist-train-images/"

    for f in os.listdir(directory + "tif/"):

        # Create directory names

        tif_dir = directory + "tif/" + f

        json_dir = directory + "json/" + f[:-4] + ".json"

        txt_dir = directory + "txt/" + f[:-4] + ".txt"

        # Get pixels

        digit_image = Image.open(tif_dir)

        pixels = []

        for i in range(28):

            for j in range(28):

                pixel = digit_image.getpixel((i, j))

                pixel = 1 - (pixel / 255)

                pixels.append(pixel)

        del digit_image

        # Save to json

        json_file = open(json_dir, "w")

        data = {"pixel": pixels}

        json.dump(data, json_file)

        json_file.close()

        # Save to text

        text_file = open(txt_dir, "w")

        for x in pixels:

            text_file.write(str(x) + "\n")

        text_file.close()

    directory = "data/mnist-test-images/"

    for f in os.listdir(directory + "tif/"):

        # Create directory names

        tif_dir = directory + "tif/" + f

        json_dir = directory + "json/" + f[:-4] + ".json"

        txt_dir = directory + "txt/" + f[:-4] + ".txt"

        # Get pixels

        digit_image = Image.open(tif_dir)

        pixels = []

        for i in range(28):

            for j in range(28):

                pixel = digit_image.getpixel((i, j))

                pixel = 1 - (pixel / 255)

                pixels.append(pixel)

        del digit_image

        # Save to json

        json_file = open(json_dir, "w")

        data = {"pixel": pixels}

        json.dump(data, json_file)

        json_file.close()

        # Save to text

        text_file = open(txt_dir, "w")

        for x in pixels:

            text_file.write(str(x) + "\n")

        text_file.close()


if (__name__ == "__main__"):

    main()
