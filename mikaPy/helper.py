import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import random

# from fake import makeFakes
def showGrid(tiles_to_show, use_title=False, title=[]):
    size = tiles_to_show.shape[0]
    columns = int(math.sqrt(size)) + 1
    rows = int(size / columns) + 1
    fig = plt.figure(figsize=(8, 8))
    for i in range(size):
        if use_title:
            fig.add_subplot(rows, columns, i + 1).set_title(str(title[i]))
        else:
            fig.add_subplot(rows, columns, i + 1)
        plt.imshow(tiles_to_show[i])
    plt.show()


def getCoords(imSize, partSize, pad=0):
    l = int(imSize / partSize)
    h = int(imSize / partSize)
    nb = l * h
    coords = []
    for _l in range(l):
        for _h in range(h):
            color = (int(random.randint(0, 255)), int(random.randint(0, 255)), int(random.randint(0, 255)) )
            x = _l * partSize
            y = _h * partSize
            coords.append([x + pad, y + pad, partSize, partSize, color])
    return np.array(coords)


def imageToTiles(img, sizes, resize):
    h = img.shape[0]
    all = []
    coords = []
    c = 1
    for size in sizes:
        coords.append(getCoords(img.shape[0], size))
        ###entier
        parts = np.array(split_image(img, [size, size])).astype('float32')
        index = 0
        for part in parts:
            _part = cv2.resize(np.array(part), (resize, resize))
            _part = cv2.cvtColor(_part, cv2.COLOR_BGR2GRAY)
            all.append(_part)
            # coords.append(np.array([size, index]))
            index += 1
        ###padding
        pad = 0
        if size % 2 == 0:
            demi = int(size / 2)
            pad = demi
            _img = img[demi:-demi, demi:-demi]
        else:
            demi = int((size + 1) / 2)
            pad = demi - 1
            _img = img[demi - 1:-demi, demi - 1:-demi]
        coords.append(getCoords(_img.shape[0], size, pad))
        parts = np.array(split_image(_img, [size, size])).astype('float32')
        index = 0
        for part in parts:
            _part = cv2.resize(np.array(part), (resize, resize))
            _part = cv2.cvtColor(_part, cv2.COLOR_BGR2GRAY)
            all.append(_part)
            # coords.append(np.array([size, index]))
            index += 1

    return np.array(all, dtype=np.float32) / 255, np.array(np.vstack(coords))


def split_image(image3, tile_size):
    image_shape = np.shape(image3)
    tile_rows = np.reshape(image3, [image_shape[0], -1, tile_size[1], image_shape[2]])
    serial_tiles = np.transpose(tile_rows, [1, 0, 2, 3])
    return np.reshape(serial_tiles, [-1, tile_size[1], tile_size[0], image_shape[2]])


def getMnist(nbr, numberFind):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    mask_train = np.isin(y_train, numberFind)
    mask_test = np.isin(y_test, numberFind)

    x_train = x_train[mask_train]
    y_train = y_train[mask_train]
    x_test = x_test[mask_test]
    y_test = y_test[mask_test]

    x_train = x_train[:nbr]
    y_train = y_train[:nbr]

    x_test = x_test[:int(nbr / 10)]
    y_test = y_test[:int(nbr / 10)]

    return (x_train, y_train), (x_test, y_test)


# def getFakeMnist(nbr, fake, numberFind):
#     x_fake = makeFakes(fake, 8, 10, 220)
#     x_fake_test = makeFakes(int(fake / 10), 8, 10, 220)
#
#     (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#     mask_train = np.isin(y_train, numberFind)
#     mask_test = np.isin(y_test, numberFind)
#
#     x_train = x_train[mask_train]
#     y_train = y_train[mask_train]
#     x_test = x_test[mask_test]
#     y_test = y_test[mask_test]
#
#     x_train = x_train[:nbr]
#     y_train = y_train[:nbr]
#     x_train[:fake] = x_fake
#     y_train[:fake] = int(10)
#     x_train, y_train = melange2Arrays(x_train, y_train)
#
#     x_test = x_test[:int(nbr / 10)]
#     y_test = y_test[:int(nbr / 10)]
#     x_test[:int(fake / 10)] = x_fake_test
#     y_test[:int(fake / 10)] = int(10)
#     x_test, y_test = melange2Arrays(x_test, y_test)
#
#     return (x_train, y_train), (x_test, y_test)


def melange2Arrays(A, B):
    indexs = np.arange(len(A))
    np.random.shuffle(indexs)
    return A[indexs], B[indexs]


def calculeSize(cap, coef):
    h = 0
    w = 0
    if cap.isOpened():
        width = cap.get(3)  # float
        height = cap.get(4)  # float
        h = height / coef
        w = width / coef
    return h, w


def resizeByCoef(frame, coef):
    _frame = np.copy(frame)
    h = _frame.shape[0] * coef
    w = _frame.shape[1] * coef
    _frame = cv2.resize(_frame, (int(w), int(h)))
    return _frame


def resizeByHauteur(frame, hauteur):
    _frame = np.copy(frame)
    coef = hauteur / _frame.shape[0]
    h = _frame.shape[0] * coef
    w = _frame.shape[1]
    _frame = cv2.resize(_frame, (int(w), int(h)))
    return _frame


def resizeByLargeurRatio(frame, largeur):
    coef = largeur / frame.shape[1]
    h = frame.shape[0] * coef
    w = frame.shape[1] * coef
    return cv2.resize(frame, (int(w), int(h)))


def resizeByHauteurRatio(frame, hauteur):
    coef = hauteur / frame.shape[0]
    h = frame.shape[0] * coef
    w = frame.shape[1] * coef
    return cv2.resize(frame, (int(w), int(h)))


def resizeByLargeur(frame, largeur):
    w = frame.shape[1] * (largeur / frame.shape[1])
    return cv2.resize(frame, (int(w), int(frame.shape[0])))


def resizeAndPad(img, size, padColor=0):
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA
    else:  # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w / h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1:  # horizontal image
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:  # vertical image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:  # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor,
                                              (list, tuple, np.ndarray)):  # color image but only one color provided
        padColor = [padColor] * 3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT,
                                    value=padColor)

    return scaled_img


def removeHPad(img, pad=10):
    return np.copy(img)[pad:-pad]


def inSquare(img, size, color=0):
    mask = np.zeros((size, size), np.uint8)
    h = img.shape[0]
    w = img.shape[1]
    y = int((size - h) / 2)
    x = int((size - w) / 2)

    mask[y:y + h, x:x + w] = img
    return mask


def coins(lineHaut, thres=100):
    a = [0, 0]
    i = 0
    tot = 0
    size = 0
    for p in lineHaut:
        if p < thres:
            a[i] += 1
        elif p > thres:
            i = 1
            size += 1
        tot += 1
    res = ()
    if a[0] > a[1]:  ### cest apres
        res = (a[0], tot)
    elif a[0] < a[1]:  ### cest avant
        res = (0, tot - a[1])
    else:
        res = (a[0], tot - a[1])
    return res, size


def perspective(lineHaut, lineBas):
    haut, sh = coins(lineHaut)
    bas, sb = coins(lineBas)
    return max(sh, sb), haut, bas


def relarge(image, plusGrandeLargeur, color=0):
    mask = np.zeros((image.shape[0], plusGrandeLargeur), np.uint8)
    mask[:, :] = color
    largeur = image.shape[1]
    decal = plusGrandeLargeur - largeur
    mask[:, decal:] = image
    return mask


def digit(image, sizeSegment):
    DIGITS_LOOKUP = {
        (1, 1, 1, 0, 1, 1, 1): 0,
        (0, 0, 1, 0, 0, 1, 0): 1,
        (1, 0, 1, 1, 1, 0, 1): 2,
        (1, 0, 1, 1, 0, 1, 1): 3,
        (0, 1, 1, 1, 0, 1, 0): 4,
        (1, 1, 0, 1, 0, 1, 1): 5,
        (1, 1, 0, 1, 1, 1, 1): 6,
        (1, 0, 1, 0, 0, 1, 0): 7,
        (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 1, 1, 1, 0, 1, 1): 9
    }
    image[image > 100] = 255
    image[image < 100] = 0
    h, w = image.shape[:2]
    centerH = int(h / 2)
    segB = np.copy(image)[h - sizeSegment: h, :]
    segBG = np.copy(image)[centerH:h, 0:sizeSegment]
    segBD = np.copy(image)[centerH:h, -sizeSegment:]
    segH = np.copy(image)[0: sizeSegment, 0: w]
    segHG = np.copy(image)[0: centerH, 0: sizeSegment]
    segHD = np.copy(image)[0: centerH, -sizeSegment:]
    segC = np.copy(image)[centerH - int(sizeSegment / 2): centerH + int(sizeSegment / 2), :]
    A = np.mean(segH) > 128
    B = np.mean(segHG) > 128
    C = np.mean(segHD) > 128
    D = np.mean(segC) > 128
    E = np.mean(segBG) > 128
    F = np.mean(segBD) > 128
    G = np.mean(segB) > 128
    tpl = tuple(np.array([A, B, C, D, E, F, G]).astype(int))
    resultat = -1
    # print(tpl)
    if tpl in DIGITS_LOOKUP:
        resultat = DIGITS_LOOKUP[tpl]
        # print("resultat", resultat)
    return resultat
    # return segH, segHG, segHD, segC, segB, segBG, segBD


if __name__ == '__main__':
    im_full = cv2.imread("./823.png", 1)
    tiles, coords = imageToTiles(im_full, [28, 35, 70], 28)
    print(tiles.shape)
    showGrid(tiles)
