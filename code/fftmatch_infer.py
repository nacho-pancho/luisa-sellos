#!/usr/bin/env python3
#
# -*- coding: utf-8 -*-
"""
Plantilla de archivo para trabajar en los distintos proyectos de LUISA
Esta plantilla no hace mas que abrir una lista de archivos, leerlos uno
por uno y guardar una copia en un directorio especificado.

Los requisitos mínimos para correr este script es tener instalados
los siguientes paquetes de Python 3.
Se recomienda utilizar el manejador de paquetes de Python3, pip3:

numpy
pillow 
matplotlib

Se recomienda también el siguiente paquete:

scipy

@author: nacho
"""
#
# # paquetes base de Python3 (ya vienen con Python)
#
import os.path
import time
import argparse
import multiprocessing
import functools
import sqlite3
import base64
import matplotlib.pyplot as plt
import scipy.signal as dsp


#
# bibliotecas adicionales necesarias
#
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage, misc
#from console_progressbar import ProgressBar # pip3 install console-progressbar
from skimage import transform # pip3 install scikit-image

#---------------------------------------------------------------------------------------

def imread(fname):
    img = Image.open(fname)
    if not fname.endswith('tif'):
        return np.asarray(img,dtype=np.uint8)
    if not 274 in img.tag_v2:
        return np.asarray(img,dtype=np.uint8)
    if img.tag_v2[274] != 8: # regression bug in PILLOW for TIFF images
        return np.asarray(img, dtype=np.uint8)
    img = img.rotate(-90, resample=Image.NEAREST,expand=True,fillcolor=1)
    return np.asarray(img,dtype=np.uint8)

#---------------------------------------------------------------------------------------

def match_single_stamp(image,stamp):
    #
    # factores de normalizacion
    #
    G = dsp.correlate(image, stamp, method="fft", mode='same')
    return int(np.round(100*np.max(G)))


#---------------------------------------------------------------------------------------

if __name__ == '__main__':
    #
    # ARGUMENTOS DE LINEA DE COMANDOS
    #
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--imgdir", type=str, default="/luisa/originales",
      help="path prefix  where to find files")
    ap.add_argument("-l","--list", type=str, default="",
      help="text file where input files are specified")
    ap.add_argument("-s", "--stampdb", type=str, default="stamps.db",
      help="stamps db")
    ap.add_argument("-o","--matchdb", type=str, default="matches.db",
      help="where to store results")
    args = vars(ap.parse_args())
    #
    # INICIALIZACION
    #
    imgdir = args["imgdir"]

    list_file  = args["list"]
    if len(list_file) == 0:
        print("ERROR: must specify list file")
        exit(1)

    stampdbfile = args["stampdb"]
    if len(stampdbfile) == 0:
        print("ERROR: must specify stamps file")
        exit(1)
    stamps_db = sqlite3.connect(stampdbfile)

    matchdbfile = args["matchdb"]
    if len(matchdbfile) == 0:
        print("ERROR: must specify match db file")
        exit(1)
    match_db = sqlite3.connect(matchdbfile)

    #
    # abrimos lista de archivos
    # la lista es un archivo de texto con un nombre de archivo
    # en cada linea
    #
    all_scores = list()
    with open(list_file) as fl:
        t0 = time.time()
        nimage = 0
        #
        # repetimos una cierta operación para cada archivo en la lista
        #
        for relfname in fl:
            relfname = relfname.split('.')[0]+".tif"
            #
            # proxima imagen
            #
            nimage = nimage + 1
            #
            # nombres de archivos y directorios de entrada y salida
            #
            relfname = relfname.rstrip('\n')
            reldir,fname = os.path.split(relfname)
            fbase,fext = os.path.splitext(fname)
            print(f'#{nimage} image={fbase}:')
            #
            # leemos imagen
            #
            input_fname = os.path.join(imgdir,relfname)
            image = imread(input_fname)
            #
            # negate (because we want black ink to be 1 and white backgrond to be 0
            #
            image = 1-image
            #
            # scale image to 1/4
            #
            image = transform.rescale(image.astype(np.float), 0.25, order=3, mode='constant', cval=1, anti_aliasing=True)
            tpl = np.ones((256, 256))
            norm = dsp.correlate(image ** 2, tpl, method='fft', mode='same')
            norm = np.sqrt(np.maximum(norm, 1e-5))
            norm_image = image / norm
            print(np.max(norm_image))
            #
            # pre-normalize
            #
            h, w = image.shape
            #
            # now compare to each stamp in the db
            #
            cursor = stamps_db.execute('SELECT hash FROM stamp')
            all_stamp_hashes = list(cursor.fetchall())
            n = len(all_stamp_hashes)
            for i,row in enumerate(all_stamp_hashes):
                stamp_hash = row[0]
                cursor = stamps_db.execute('SELECT * FROM cache WHERE stamp_hash=:stamp_hash',{'stamp_hash':stamp_hash})
                c =  cursor.fetchone()
                print('\tstamp',i,'/',n)
                rotated_stamps = list()
                while c is not None:
                    #print('\t\trotation')
                    hs, ws, pix64 = c[1], c[2], c[-1]
                    pixbytes = base64.b64decode(pix64)
                    stamp = 255-np.reshape(np.frombuffer(pixbytes,dtype=np.uint8),(hs,ws))
                    stamp = stamp * (1 / np.linalg.norm(stamp.ravel()))
                    rotated_stamps.append(stamp)
                    c = cursor.fetchone()
                hs, ws = rotated_stamps[0].shape
                correlation = np.zeros(norm_image.shape)
                pool = multiprocessing.Pool()
                scores = pool.map(functools.partial(match_single_stamp, norm_image), rotated_stamps)
                score = np.max(scores)
                print('score',score)
                pool.close()
                pool.join()
        #
        # fin para cada archivo en la lista
        #

    #
    # fin main
    #
#---------------------------------------------------------------------------------------
