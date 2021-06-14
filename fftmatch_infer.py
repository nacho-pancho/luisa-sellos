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
# import os
import os.path
import sys
import time
import argparse
import math
import csv
import multiprocessing
import functools

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

verbose = False

#---------------------------------------------------------------------------------------

def imread(fname):
    img = Image.open(fname)
    if not fname.endswith('tif'):
        return 1-np.asarray(img,dtype=np.uint8)
    if not 274 in img.tag_v2:
        return 1-np.asarray(img,dtype=np.uint8)
    if img.tag_v2[274] == 8: # regression bug in PILLOW for TIFF images
        img = img.rotate(-90, resample=Image.NEAREST,expand=True,fillcolor=1)
    return 1-np.asarray(img,dtype=np.uint8)

#---------------------------------------------------------------------------------------

def imsave(fname,img):
    aux = img.copy()
    aux -= np.min(aux)
    aux = np.uint8(255*aux/max(1e-10,np.max(aux)))
    Image.fromarray(aux).save(fname)

#---------------------------------------------------------------------------------------

def reducir(a,razon):
    ares =  transform.rescale(a.astype(np.float),razon,order=1,mode='constant',cval=0,anti_aliasing=True)
    return ares

#---------------------------------------------------------------------------------------

def ampliar(a,shape):
    return  transform.resize(a,shape,mode='constant',cval=0,anti_aliasing=True)

#---------------------------------------------------------------------------------------

def correlate(img,sello):
    return dsp.correlate( img, sello, method="fft",mode='same' ) 

#---------------------------------------------------------------------------------------

def match_single_stamp(image_data,scale,rotated_stamp):
    #
    # factores de normalizacion
    #
    Is,Isn  = image_data
    m,n = Is.shape
    best_score = 0
    best_angle = 0
    best_i = 0
    best_j = 0
    #best_G = np.zeros(Is.shape)
    for a in rotated_stamp.keys():
        ssr = rotated_stamp[a]
        G = correlate( Is, ssr ) / Isn 
        limax = np.argmax( G )
        imax  = limax // n
        jmax  = limax - imax*n
        gmax = G[imax,jmax]
        if best_score < gmax:
            best_score = gmax
            #best_angle = a
            #best_i     = imax
            #best_j     = jmax
            #best_G     = G.copy()
    #if verbose:
    #    print(f"\tscale {s:7.5f} angle {best_angle:5.2f} maximum {best_score:5.3f} at ({best_i:5d},{best_j:5d})")
    #limax = np.argmax( best_G )
    #imax  = limax // base_shape[1]
    #jmax  = limax - imax*base_shape[1]
    #best_score = best_G[imax,jmax]
    return best_score
    
#---------------------------------------------------------------------------------------

def match_many_stamps(I,stamps,args,scale=0.25):
    cachedir = args["cachedir"]
    imgname  = args["name"]
    score_cache = os.path.join(cachedir,f"{imgname}-scores.npy")
    if os.path.exists(score_cache):
        scores = np.load(score_cache)
        return scores

    if not os.path.exists(cachedir):
        os.makedirs(cachedir)
    maxw = 0 
    maxh = 0
    angles = np.arange(-3,3.5,0.5)
    #
    # tamaño comun para todos los sellos
    #
    for i in range(len(stamps)):
        if stamps[i].shape[0] > maxh:
            maxh = stamps[i].shape[0]
        if stamps[i].shape[1] > maxw:
            maxw = stamps[i].shape[1]
    # precalcular imagenes escaladas y normalizadas
    #
    # usamos una convolucion para estimar la norma 2 de cada patch de la imagen
    # es la raiz de la suma del cuadrado de los pixeles en cada patch
    #
    maxhs = int(np.ceil(1.1*maxh*scale))
    maxws = int(np.ceil(1.1*maxw*scale))
    plantilla1 = np.ones((maxhs,maxws))
    fcache = os.path.join(cachedir,f"{imgname}-scale{scale:5.3f}.npy")
    if not os.path.exists(fcache):
        Is =  reducir( I, scale )
        Isn  = correlate( Is**2, plantilla1 )
        Isn = np.sqrt(np.maximum(Isn,1e-16))
        np.save(fcache,Is)            
        fcache = os.path.join(cachedir,f"{imgname}-scale{scale:5.3f}.jpg")
        imsave(fcache,Is)
        fcache = os.path.join(cachedir,f"{imgname}-scale{scale:5.3f}-norm.npy")
        np.save(fcache,Isn)            
        fcache = os.path.join(cachedir,f"{imgname}-scale{scale:5.3f}-norm.jpg")
        imsave(fcache,Isn)
        image_data = (Is, Isn)            
    else:
        Is = np.load(fcache)
        fcache = os.path.join(cachedir,f"{imgname}-scale{scale:5.3f}-norm.npy")
        Isn = np.load(fcache)
        image_data = (Is, Isn)            
    #
    # precalcular sellos escalados, rotados y normalizados
    #
    nstamps = len(stamps)
    rotated_stamps = list()
    for i in range( nstamps ):
        #print(f"stamp {i} of {nstamps}")
        stamp = stamps[ i ]
        rotated_stamp = dict()
        for a in angles:
            aa = abs(a)
            if a < 0:
                sgn = '-'
            else:
                sgn = '+'
            fcache = os.path.join(cachedir,f"sello{i:02d}-scale{scale:5.3f}-angle{sgn}{aa:06.3f}.npy")
            if os.path.exists(fcache):
                rotated_stamp[a] = np.load(fcache)
            else:    
                ssr = reducir( ndimage.rotate( stamp, a ), scale ) 
                sh,sw = ssr.shape
                sn    = 1.0 / np.linalg.norm( ssr.ravel() )                
                aux = np.zeros((maxhs,maxws))
                ioff  = ( maxhs - sh ) // 2
                joff  = ( maxws - sw ) // 2
                #print(maxhs,ioff,sh,maxws,joff,sw)
                aux[ ioff:ioff+sh, joff:joff+sw ] = ssr*sn
                rotated_stamp[a] = aux
                np.save(fcache,aux)
                fcache = os.path.join(cachedir,f"sello{i:02d}-scale{scale:5.3f}-angle{sgn}{aa:06.3f}.png")
                imsave(fcache,aux)
        rotated_stamps.append(rotated_stamp)
    #
    # hacer la comparacion
    #
    pool = multiprocessing.Pool()
    scores = pool.map( functools.partial(match_single_stamp, image_data, scale), rotated_stamps)
    pool.close()
    pool.join()
    #if verbose:
    #    print( f"sello {i:3d} score {score:5.3f}" )
    #scores.append(score)
    np.save(score_cache,scores)
    return scores
#---------------------------------------------------------------------------------------



if __name__ == '__main__':
    #
    # ARGUMENTOS DE LINEA DE COMANDOS
    #
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--imgdir", type=str, default="/luisa/originales",
      help="path prefix  where to find files")
    ap.add_argument("-S", "--stampdir", type=str, default="/luisa/sellos_clasificados",
      help="path prefix  where to find files")
    ap.add_argument("-c", "--cachedir", type=str, default="../cache",
      help="path prefix  where to find cache files")
    ap.add_argument("-o","--outdir", type=str, default="../results",
      help="where to store results")
    ap.add_argument("-l","--list", type=str, default="",
      help="text file where input files are specified")
    ap.add_argument("-s","--stamps", type=str, default="",
      help="text file with the list of input stamp image files")
    args = vars(ap.parse_args())
    #
    # INICIALIZACION
    #
    list_file  = args["list"]
    if len(list_file) == 0:
        print("ERROR: must specify list file")
        exit(1)
    stamps_file = args["stamps"]
    if len(stamps_file) == 0:
        print("ERROR: must specify stamps file")
        exit(1)

    imgdir = args["imgdir"]
    stampdir = args["stampdir"]
    outdir = args["outdir"]
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    cachedir = args["cachedir"]
    if not os.path.exists(cachedir):
        os.makedirs(cachedir)
    #
    # cargamos sellos
    #
    sellos = list()

    with open(stamps_file) as fl:
        nimage = 0
        for relfname in fl:
            nimage = nimage+1
            relfname = relfname.rstrip('\n')
            reldir,fname = os.path.split(relfname)
            fbase,fext = os.path.splitext(fname)
            input_fname = os.path.join(stampdir,relfname)
            sello = imread(input_fname)
            sellos.append(sello)

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
            foutdir = os.path.join(outdir,reldir)
            debugdir = os.path.join(foutdir,fbase + "_debug")            
            print(f'#{nimage} image={fbase}:')
            #
            # creamos carpetas de destino si no existen
            #
            if not os.path.exists(foutdir):
                os.makedirs(foutdir)


            output_fname = os.path.join(foutdir,fname)
            input_fname = os.path.join(imgdir,relfname)
            #
            # leemos imagen
            #
            img = imread(input_fname)
            #---------------------------------------------------
            # hacer algo en el medio
            #---------------------------------------------------
            #
            args["reldir"] = reldir
            args["name"]   = fbase
            scores = match_many_stamps(img,sellos,args)
            print("\tscore:",[np.round(d,2) for d in scores])
            all_scores.append(scores)
            #---------------------------------------------------
        #
        # fin para cada archivo en la lista
        #

    #
    # fin main
    #
#---------------------------------------------------------------------------------------
