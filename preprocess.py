from PIL import Image
import numpy as np
import json
from tqdm import tqdm
import shapely
import shapely.wkt
import cv2
import os
import zipfile
import shutil
from collections import defaultdict


def damage_label():
    
    d = defaultdict(lambda:0)
    d['no-damage'] = 0
    d['minor-damage'] = 1
    d['major-damage'] = 2
    d['destroyed'] = 3
    
    return d

def create_dir(path):

    if path not in os.listdir('/'):
        os.mkdir(path)
        os.mkdir(path + '/images')
        os.mkdir(path + '/labels')
    else:
        shutil.rmtree(path)
        os.mkdir(path)
        os.mkdir(path + '/images')
        os.mkdir(path + '/labels')

def move_files(source, dest):

    folder_list = ['hold', 'test', 'train']
    subfolder_list = ['/images', '/labels']
    img_dest_dir = source + '/images'
    json_dest_dir = source + '/labels' 

    for folder in folder_list:
        for files in os.listdir('/palu/' + folder +  subfolder_list[0]):
            shutil.copy('/palu/' + folder + subfolder_list[0] + '/' + files, img_dest_dir)

    for folder in folder_list:
        for files in os.listdir('/palu/' + folder +  subfolder_list[1]):
            shutil.copy('/palu/' + folder + subfolder_list[1] + '/' + files, json_dest_dir)

    os.mkdir(dest)
    shutil.move(source, dest)

def preprocess(disaster_paths, save_processed_path):
    
    damage_intensity_encoding = damage_label()

    image_paths = []
    image_paths.extend(disaster_paths + '/' + image for image in os.listdir(disaster_paths))

    x_data = []
    y_data = []

    

    if 'processed_img' not in os.listdir('/xBD/full_palu'):
      os.mkdir(save_processed_path)
    else:
       shutil.rmtree(save_processed_path)
       os.mkdir(save_processed_path)

    for img_path in tqdm(image_paths):

        img_obj = Image.open(img_path)
        img_array = np.array(img_obj)
        height, width, color = img_array.shape

        label_path = img_path.replace('images', 'labels').replace('png', 'json')
        label_file = open(label_path)
        label_data = json.load(label_file)

        for feat in label_data['features']['xy']:
            try:
                damage_type = feat['properties']['subtype']
            except: 
                damage_type = 'no-damage'
                continue
            y_data.append(damage_intensity_encoding[damage_type])

            polygon_geom = shapely.wkt.loads(feat['wkt'])
            polygon_pts = np.array(list(polygon_geom.exterior.coords))

            xcoords = polygon_pts[:, 0]
            ycoords = polygon_pts[:, 1]

            xmin = np.min(xcoords)
            xmax = np.max(xcoords)
            ymin = np.min(ycoords)
            ymax = np.max(ycoords)

            xdiff = xmax - xmin
            ydiff = ymax - ymin

            xmin = max(int(xmin - (xdiff * 0.75)), 0)
            xmax = min(int(xmax + (xdiff * 0.75)), width)
            ymin = max(int(ymin - (ydiff * 0.75)), 0)
            ymax = min(int(ymax + (ydiff * 0.75)), height)

            poly_img = img_array[ymin:ymax, xmin:xmax, :]

            poly_uuid = feat['properties']['uid'] + '.png'
            cv2.imwrite(save_processed_path + "/" + poly_uuid, poly_img)
            x_data.append(poly_uuid)


if __name__ == "__main__":
    
    data_path = '/palu-disaster-satellite-images.zip'

    with zipfile.ZipFile(data_path, 'r') as zip_ref:
        zip_ref.extractall()
    
    create_dir(path='full_palu')
    move_files(source='full_palu', dest='xBD')
    
    preprocess(disaster_paths='/xBD/full_palu/images', save_processed_path='/xBD/full_palu/processed_img')