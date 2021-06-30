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
import argparse
import sqlite3
import base64

import scipy.signal as dsp


#
# bibliotecas adicionales necesarias
#
import numpy as np
from PIL import Image
from skimage import transform,io # pip3 install scikit-image
import hashlib

verbose = False

#---------------------------------------------------------------------------------------

def imread(fname):
    img = Image.open(fname)
    if not fname.endswith('tif'):
        return np.asarray(img,dtype=np.uint8)
    if not 274 in img.tag_v2:
        return np.asarray(img,dtype=np.uint8)
    if img.tag_v2[274] != 8:
        return np.asarray(img,dtype=np.uint8)
    # regression bug in PILLOW for TIFF images
    return img.rotate(-90, resample=Image.NEAREST,expand=True,fillcolor=0)




if __name__ == '__main__':
    #
    # ARGUMENTOS DE LINEA DE COMANDOS
    #
    ap = argparse.ArgumentParser()
    ap.add_argument("-S", "--stampdir", type=str, default="/luisa/sellos_clasificados",
      help="path prefix  where to find files")
    ap.add_argument("-c", "--dbfile", type=str, default="stamps.db",
      help="path to output SQLite3 database")
    ap.add_argument("-l","--list", type=str, default="",
      help="text file where input files are specified")
    args = vars(ap.parse_args())
    #
    # INICIALIZACION
    #
    list_file  = args["list"]
    if len(list_file) == 0:
        print("ERROR: must specify list file")
        exit(1)

    stampdir = args["stampdir"]
    dbfile  = args["dbfile"]
    angles = np.arange(-5, 5.5, 0.5)
    scales = list()
    scales.append(0.25)

    classes = set()
    #
    # create DB
    #
    db   = sqlite3.connect(dbfile)
    db.execute("CREATE TABLE class  (id integer primary key, pathname varchar not null)")
    db.execute("CREATE TABLE stamp  (id integer primary key, "
               "hash char[32], class varchar,"
               "filename varchar not null, width integer, "
               "height integer, pixels varchar )")
    #
    # first pass: scan stamps, sizes, binary hash
    #
    nstamps = 0
    nclasses = 0
    maxw = 0
    maxh = 0
    print('FIRST PASS')

    with open(list_file) as fl:
        for relfname in fl:
            nstamps += 1
            #
            # locations, filenames, etc.
            #
            relfname = relfname.rstrip('\n')
            reldir,fname = os.path.split(relfname)
            fbase,fext = os.path.splitext(fname)
            input_fname = os.path.join(stampdir,relfname)
            #
            # class (=folder name) and sample within class (=file name)
            #
            stamp_class = reldir
            stamp_sample = fbase
            #
            # hash of the pixels as 8 bit grayscale samples
            #
            stamp_img = imread(input_fname)

            h,w  = stamp_img.shape
            pix = np.packbits(stamp_img.astype(np.bool))
            hash = hashlib.sha256(pix).hexdigest()
            pix64 = base64.b64encode(pix)
            #print('input:',input_fname,end='\t')
            #print('class:',stamp_class,end='\t')
            #print('sample:',stamp_sample,end='\t')
            #print('width:',w,'height:',h,end='\t')
            #print('hash:',hash)
            #
            # store in DB
            #
            if stamp_class not in classes:
                classes.add(stamp_class)
                db.execute('INSERT INTO class VALUES (:id,:pathname)',{'id':nclasses,'pathname':stamp_class})
                nclasses += 1
            db.execute('INSERT INTO stamp VALUES (:id,:hash,:filename,:class,:width,:height,:pixels)',
                    {'id':nstamps,'hash':hash,'filename':stamp_sample,'class':stamp_class,'width':w,'height':h,'pixels':pix64})
            if w > maxw:
                maxw = w
            if h > maxh:
                maxh = h
    print('NUMBER OF STAMPS:',nstamps)
    print('MAX WIDTH',maxw,'MAX HEIGHT',maxh)
    db.commit()
    #
    # create cache with rotated images
    #
    print('SECOND PASS')
    db.execute("CREATE TABLE cache (stamp_id integer, "
               "height integer, width integer, "
               "scale real, angle real, pixels varchar )")
    total_stamps = nstamps
    nstamps = 0
    with open(list_file) as fl:
        for relfname in fl:
            print(nstamps,'/',total_stamps)
            #
            # locations, filenames, etc.
            #
            relfname = relfname.rstrip('\n')
            reldir,fname = os.path.split(relfname)
            fbase,fext = os.path.splitext(fname)
            input_fname = os.path.join(stampdir,relfname)
            #
            # class (=folder name) and sample within class (=file name)
            #
            stamp_class = reldir
            stamp_sample = fbase
            stamp_img = imread(input_fname)
            pix = np.packbits(stamp_img.astype(np.bool))
            stamp_hash = hashlib.sha256(pix).hexdigest()

            stamp_img = stamp_img.astype(np.float)
            #
            #
            #
            for scale in scales:
                #smallh = int(np.power(2, np.ceil(np.log2(maxh * scale))))
                #smallw = int(np.power(2, np.ceil(np.log2(maxw * scale))))
                scaled_img = transform.rescale(stamp_img, scale, order=3, mode='constant', cval=1.0, anti_aliasing=True)
                #pad_top = (smallh - scaled_h) // 2
                #pad_bottom = smallh - scaled_h - pad_top
                #pad_left = (smallw - scaled_w) // 2
                #pad_right  = smallw - scaled_w - pad_left
                #padded_img = np.pad(scaled_img,((pad_top,pad_bottom),(pad_left,pad_right)),mode='constant',constant_values=1.0 )
                padded_img = np.pad(scaled_img,((20,20),(20,20)),mode='constant',constant_values=1)
                #padded_img = scaled_img
                for angle in angles:
                    scaled_rotated_img = transform.rotate(padded_img,angle,order=3,mode='constant',cval=1)
                    scaled_rotated_img[scaled_rotated_img < 0.0] = 0.0
                    scaled_rotated_img[scaled_rotated_img > 1.0] = 1.0
                    scaled_rotated_img = (255.0*scaled_rotated_img).astype(np.uint8)
                    pix64 = base64.b64encode(scaled_rotated_img.tobytes())
                    scaled_h, scaled_w = scaled_rotated_img.shape
                    db.execute('INSERT INTO cache VALUES (:stamp_id,:height,:width,:scale,:angle,:pixels)',
                       {'stamp_id':nstamps,'height':scaled_h,'width':scaled_w,'scale':scale,'angle':angle,'pixels':pix64})
            nstamps += 1
        db.commit()
    print('FINISHED')
    #
    # fin main
    #
#---------------------------------------------------------------------------------------
