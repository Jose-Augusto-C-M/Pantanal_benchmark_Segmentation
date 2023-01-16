import os
import numpy as np
from PIL import Image, ImageDraw
import time

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

import rasterio
from rasterio.windows import Window

import shapefile
import cv2

import argparse
from threading import Thread
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import gc

# Import necessary modules first
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import fiona
from fiona.crs import from_epsg

max_value = float(2**15)

def save_shapefile_internal(filename, results, dataset):
    newdata = gpd.GeoDataFrame()
    newdata['geometry'] = None
    contours, hierarchy = cv2.findContours(results, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    polygons_internal = []
    index = 0
    new_img = Image.fromarray(np.zeros(results.shape, dtype=np.uint8))
    draw = ImageDraw.Draw(new_img)
    num = 0
    while index != -1:
        #print(index)
        cnt = contours[index]
        cnt = np.squeeze(cnt)
        if cnt.shape[0] <= 5:
            index = hierarchy[0, index, 0]
            continue
        poly_ext = []
        points = []
        for x,y in cnt:
            points.append((x, y))
            position = [dataset.xy(y, x)]
            poly_ext.append((position[0][0], position[0][1]))
        polygons.append(poly_ext)
        draw.polygon((points), fill=255)
        child = hierarchy[0, index, 2]
        internals = []
        if child != -1:
            while child != -1:
                cnt = contours[child]
                cnt = np.squeeze(cnt)
                poly = []
                points = []
                for x,y in cnt:
                    points.append((x, y))
                    position = [dataset.xy(y, x)]
                    poly.append((position[0][0], position[0][1]))
                internals.append(poly)
                draw.polygon((points), fill=0)
                child = hierarchy[0, child, 0]
        if len(internals) > 0:
            newdata.loc[num, 'geometry'] = Polygon(poly_ext, internals)
        else:
            newdata.loc[num, 'geometry'] = Polygon(poly_ext)
        num += 1
        polygons_internal.append(internals)
        index = hierarchy[0, index, 0]
    new_img.save('polygons.png')
    # Create an empty geopandas GeoDataFrame
    #newdata.crs = from_epsg(4326)
    # Determine the output path for the Shapefile
    # Write the data into that Shapefile
    newdata.to_file(filename + '.shp')
    #w = shapefile.Writer(filename + '.shp')
    #w.field('name', 'C')
    #w.poly(polygons)
    #w.record('swimming pools')
    #w.close()

def save_shapefile(filename, results, dataset):
    contours, _ = cv2.findContours(results, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        cnt = np.squeeze(cnt)
        print(cnt.shape)
        if cnt.shape[0] <= 5: continue
        poly = []
        for x,y in cnt:
            position = [dataset.xy(y, x)]
            poly.append((position[0][0], position[0][1]))
        polygons.append(poly)

    w = shapefile.Writer(filename)
    w.field('name', 'C')
    w.poly(polygons)
    w.record('swimming pools')
    w.close()

def get_bands(tif):
    dataset = rasterio.open(tif)
    band_b, band_g, band_r, band_nir = dataset.read(1), dataset.read(2), dataset.read(3), dataset.read(4)
    # band_b, band_g, band_r = dataset.read(1), dataset.read(2), dataset.read(3)
    not_valid_0 = np.logical_and(np.logical_and(band_r == 0, band_g == 0), band_b == 0)
    valid = np.logical_not(not_valid_0)
    band_r = band_r.astype(np.float32)
    band_g = band_g.astype(np.float32)
    band_b = band_b.astype(np.float32)
    band_nir = band_nir.astype(np.float32)

    min_r, max_r = np.min(band_r[valid]), np.max(band_r[valid])
    min_g, max_g = np.min(band_g[valid]), np.max(band_g[valid])
    min_b, max_b = np.min(band_b[valid]), np.max(band_b[valid])
    min_nir, max_nir = np.min(band_nir[valid]), np.max(band_nir[valid])

    band_ndvi = np.zeros((band_r.shape[:2]), dtype=np.float32)
    band_ndvi[valid] = (band_nir[valid]-band_r[valid]) / (band_nir[valid]+band_r[valid])
    min_ndvi, max_ndvi = np.min(band_ndvi[valid]), np.max(band_ndvi[valid])
    if np.sum(np.isnan(band_ndvi)) > 0:
        print('Valor NaN')
        sys.exit(0)
        #band_ndvi[np.isnan(band_ndvi)] = 0.

    band_ndviv = np.zeros((band_r.shape[:2]), dtype=np.float32)
    band_ndviv[valid] = (band_g[valid]-band_r[valid]) / (band_g[valid]+band_r[valid])
    min_ndviv, max_ndviv = np.min(band_ndviv[valid]), np.max(band_ndviv[valid])
    if np.sum(np.isnan(band_ndviv)) > 0:
        print('Valor NaN')
        sys.exit(0)
        #band_ndviv[np.isnan(band_ndviv)] = 0.

    # band_r[valid] = (((band_r[valid]-min_r) / (max_r-min_r))*255).astype(np.uint8)
    # band_g[valid] = (((band_g[valid]-min_g) / (max_g-min_g))*255).astype(np.uint8)
    # band_b[valid] = (((band_b[valid]-min_b) / (max_b-min_b))*255).astype(np.uint8)
    # band_nir[valid] = (((band_nir[valid]-min_nir) / (max_nir-min_nir))*255).astype(np.uint8)
    # band_ndvi[valid] = (((band_ndvi[valid]-min_ndvi) / (max_ndvi-min_ndvi))*255).astype(np.uint8)
    # band_ndviv[valid] = (((band_ndviv[valid]-min_ndviv) / (max_ndviv-min_ndviv))*255).astype(np.uint8)

    band_r[valid] = ((band_r[valid] / max_value)*255).astype(np.uint8)
    band_g[valid] = ((band_g[valid] / max_value)*255).astype(np.uint8)
    band_b[valid] = ((band_b[valid] / max_value)*255).astype(np.uint8)
    band_nir[valid] = ((band_nir[valid] / max_value)*255).astype(np.uint8)
    band_ndvi[valid] = ((band_ndvi[valid] / max_value)*255).astype(np.uint8)
    band_ndviv[valid] = ((band_ndviv[valid] / max_value)*255).astype(np.uint8)

    return band_r, band_g, band_b, band_nir, band_ndvi, band_ndviv, dataset, valid
    # return band_r, band_g, band_b, dataset, valid


def get_data(img_path, model_path, model_config, w=512, h=512, step=16, save_img=False):
    global model

    start = time.time()

    if not (img_path.endswith('.tif') or img_path.endswith('.jp2')):
        print('Image is not tif')
        return None

    filename = os.path.splitext(os.path.split(img_path)[1])[0]
    print(filename)

    band_r, band_g, band_b, band_nir, band_ndvi, band_ndviv, dataset, valid = get_bands(img_path)
    img_w = dataset.width
    img_h = dataset.height

    results = np.zeros((img_h, img_w), dtype=np.uint8)

    for x in range(0, img_h, step):
        print('%d of %d' % (x, img_h))
        for y in range(0, img_w, step):
            if x+h >= img_h: x = img_h-h
            if y+w >= img_w: y = img_w-w
            img = np.zeros((w,h,4), dtype=np.float32)
            # img[:,:,0] = band_b[x:x+h, y:y+w]
            # img[:,:,1] = band_nir[x:x+h, y:y+w]
            # img[:,:,2] = band_ndviv[x:x+h, y:y+w]

            img[:,:,0] = band_r[x:x+h, y:y+w]
            img[:,:,1] = band_g[x:x+h, y:y+w]
            img[:,:,2] = band_b[x:x+h, y:y+w]
            img[:,:,3] = band_nir[x:x+h, y:y+w]
            #if np.sum(valid[x:x+h, y:y+w]) < w*h*0.1: continue
            #img = img[:, :, [2, 1, 0]]
            result = inference_segmentor(model, img)[0]

            x1 = x+int(0)
            y1 = y+int(0)
            x2 = x+int(w)
            y2 = y+int(h)
            results[x1:x2, y1:y2] = np.maximum(results[x1:x2, y1:y2], result[x1-x:x2-x, y1-y:y2-y])

    Image.fromarray((results*100)).save(filename+'.png')

    save_shapefile_internal(filename, results, dataset)

    end = time.time()
    print('\n\nTempo')
    print(end - start)
    print('\n\n')

    return end-start

def main(model_path, model_config, img_path, num_workers=5):
    imgs = [os.path.join(img_path, el) for el in os.listdir(img_path) if el.endswith('.tif')]
    
    global model
    model = init_segmentor(model_config, model_path, device='cuda:0')


    with concurrent.futures.ThreadPoolExecutor(max_workers = num_workers) as executor:
        for i in range(0,len(imgs), num_workers):
            lim = i+num_workers if i+num_workers<len(imgs) else len(imgs)
            future_to_shps = {executor.submit(get_data, img, model_path, model_config, step=128, save_img=True): img for img in imgs[i:i+lim]}

            for future in concurrent.futures.as_completed(future_to_shps):
               img = future_to_shps[future]
               try:
                  data = future.result()
               except Exception as exc:
                  print('%r generated an exception: %s' % (img, exc))
               else:
                  print('Source image was %r. Shapefile generated in %fs.' % (img, data))


    # get_data(img_path, model_path, model_config, step=128, save_img=True)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/home/wesley/max/mmsegmentation/work_dirs/segformer_mit-b0_8x1_256x256_40k_pantanal_fold1/best_mIoU_iter_40000.pth', type=str)
    parser.add_argument('--model_config', default='/home/wesley/max/mmsegmentation/work_dirs/segformer_mit-b0_8x1_256x256_40k_pantanal_fold1/segformer_mit-b0_8x1_256x256_40k_pantanal.py', type=str)
    parser.add_argument('--img_path', default='./imgs/', type=str)
    parser.add_argument('--num_threads', default=5, type=int)

    args = parser.parse_args()

    model_path = args.model_path
    model_config = args.model_config
    img_path = args.img_path
    num_threads = args.num_threads

    main(model_path, model_config, img_path, num_threads)
    
    
