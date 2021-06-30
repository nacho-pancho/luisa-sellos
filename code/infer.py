#!/usr/bin/env python3
#
# -*- coding: utf-8 -*-
"""

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
import hashlib

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

def init_match_db(matchdbfile):
    db = sqlite3.connect(matchdbfile)
    db.execute("CREATE TABLE image (id integer primary key, hash char[32], pathname varchar)")
    db.execute("CREATE TABLE match (image_id integer, stamp_id integer, score real ) ")
    db.execute("CREATE TABLE class (id integer primary key, "
               "pathname varchar not null)")
    db.execute("CREATE TABLE stamp (id integer primary key, "
               "hash char[32],"
               "class integer,"
               "filename varchar, "
               "width integer, "
               "height integer, "
               "pixels varchar )")
    return db

#---------------------------------------------------------------------------------------

if __name__ == '__main__':
    #
    # ARGUMENTOS DE LINEA DE COMANDOS
    #
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--imgdir", type=str, default="/datos/luisa/originales",
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
    match_db = init_match_db(matchdbfile)

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
            image_id = nimage - 1
            image_hash = hashlib.sha256(image.tobytes()).hexdigest()
            match_db.execute("INSERT INTO image VALUES (:id,:hash,:path)",
                             {'id':image_id, 'hash': image_hash, 'path': relfname})
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
            #
            # pre-normalize
            #
            h, w = image.shape
            #
            # now compare to each stamp in the db
            #
            cursor = stamps_db.execute('SELECT id,height,width FROM stamp')
            for i,row in enumerate(cursor.fetchall()):
                stamp_id = row[0]
                cursor = stamps_db.execute('SELECT * FROM cache WHERE stamp_id=:stamp_id',
                                           {'stamp_id':stamp_id})
                c =  cursor.fetchone()
                print('\tstamp',i)
                rotated_stamps = list()
                while c is not None:
                    hs, ws = c[1], c[2]
                    #print('\t\trotation')
                    pix64 = c[-1]
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
                match_db.execute("INSERT INTO match VALUES (:stamp_id,:image_id,:score)",
                                 {'stamp_id':stamp_id,'image_id':image_id,'score':score})
                pool.close()
                pool.join()
            match_db.commit()
        #
        # fin para cada archivo en la lista
        #

    #
    # fin main
    #
#---------------------------------------------------------------------------------------
