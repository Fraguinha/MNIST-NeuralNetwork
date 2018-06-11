from PIL import Image
import os


def main():

    directory = "data/mnist-train-images/"

    for f in os.listdir(directory + "tif/"):

        tif = directory + "tif/" + f

        txt = directory + "txt/" + f[:-4] + ".txt"

        text_file = open(txt, "w")

        digit_image = Image.open(tif)

        for i in range(28):

            for j in range(28):

                coordenate = (i, j)

                pixel = digit_image.getpixel(coordenate)

                pixel = pixel / 255

                pixel = 1 - pixel

                text_file.write(str(pixel) + "\n")

                del coordenate

        del digit_image

        text_file.close()

    directory = "data/mnist-test-images/"

    for f in os.listdir(directory + "tif/"):

        tif = directory + "tif/" + f

        txt = directory + "txt/" + f[:-4] + ".txt"

        text_file = open(txt, "w")

        digit_image = Image.open(tif)

        for i in range(28):

            for j in range(28):

                coordenate = (i, j)

                pixel = digit_image.getpixel(coordenate)

                pixel = pixel / 255

                pixel = 1 - pixel

                text_file.write(str(pixel) + "\n")

                del coordenate

        del digit_image

        text_file.close()


if (__name__ == "__main__"):

    main()
