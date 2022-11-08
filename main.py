from collections import defaultdict
import os
import shutil
import json

import pandas as pd
import numpy as np

from PIL import Image, ImageDraw
import shapely
import shapely.wkt
import cv2

import tensorflow as tf
import keras.preprocessing



def load_model(building_model_path):
    model = tf.keras.models.load_model(building_model_path)
    return model

def process_img(img_array, polygon_pts, scale_pct):
    
    height, width, _ = img_array.shape

    xcoords = polygon_pts[:, 0]
    ycoords = polygon_pts[:, 1]
    xmin, xmax = np.min(xcoords), np.max(xcoords)
    ymin, ymax = np.min(ycoords), np.max(ycoords)

    xdiff = xmax - xmin
    ydiff = ymax - ymin

    xmin = max(int(xmin - (xdiff * scale_pct)), 0)
    xmax = min(int(xmax + (xdiff * scale_pct)), width)
    ymin = max(int(ymin - (ydiff * scale_pct)), 0)
    ymax = min(int(ymax + (ydiff * scale_pct)), height)

    return img_array[ymin:ymax, xmin:xmax, :]

def damage_intensity_encoding():
    damage_intensity_encoding = defaultdict(lambda: 0)
    damage_intensity_encoding['destroyed'] = 3
    damage_intensity_encoding['major-damage'] = 2
    damage_intensity_encoding['minor-damage'] = 1
    damage_intensity_encoding['no-damage'] = 0

    return damage_intensity_encoding

def create_csv(img_path, json_path, output_process_img_poly_path, output_csv):
    output_dir = output_process_img_poly_path

    x_data = []
    y_data = []

    img_obj = Image.open(img_path)
    img_array = np.array(img_obj)

    label_data = json.load(open(json_path))

    for feat in label_data['features']['xy']:
        try:
            damage_type = feat['properties']['subtype']
        except:
            damage_type = "no-damage"
            continue

        y_data.append(damage_intensity_encoding()[damage_type])
        poly_uuid = feat['properties']['uid'] + ".png"

        polygon_geom = shapely.wkt.loads(feat['wkt'])
        polygon_pts = np.array(list(polygon_geom.exterior.coords))
        poly_img = process_img(img_array, polygon_pts, 0.8)

        cv2.imwrite(output_dir + "/" + poly_uuid, poly_img)

        x_data.append(poly_uuid)
                
    df_array = {'uuid':x_data, 'labels':y_data}
    df = pd.DataFrame(data = df_array)
    df.to_csv(output_csv)

    return df

def create_prediction(img_path, json_path, building_model_path, predictions_json_path, output_process_img_poly_path, output_csv):
    df = create_csv(img_path, json_path, output_process_img_poly_path, output_csv)
    df['labels'] = df['labels'].apply(str)

    gen = keras.preprocessing.image.ImageDataGenerator(
                                rescale=1/255.)
    output_dir = output_process_img_poly_path
    validation_gen = gen.flow_from_dataframe(dataframe = df,
                                    directory = output_dir,
                                    x_col = 'uuid',
                                    y_col = 'labels',
                                    batch_size = 64,
                                    shuffle = False,
                                    seed = 123,
                                    class_mode="categorical",
                                    verbose = 1,
                                    target_size=(128, 128))

    predictions = load_model(building_model_path).predict(validation_gen)
    val_pred = np.argmax(predictions, axis = -1)
    
    if not os.path.exists(predictions_json_path):
        shutil.copy(json_path, predictions_json_path)
        
    df_prediction = pd.DataFrame({'uuid':df['uuid'], 'labels':val_pred})
    
    json_pred = json.load(open(predictions_json_path))

    for i in range(df_prediction.shape[0]):
        json_pred['features']['xy'][i]['properties']['subtype'] = df_prediction['labels'][i]
    
    return json_pred

def blank_image():
    img = Image.new("RGB", (1024, 1024), (255, 255, 255))
    return img

def mask_colors():
    colors = {
        0: (0, 255, 0, 50),
        1: (255, 0, 0, 50),
    }
    return colors

def predicted_mask(img_path, json_path, building_model_path, predictions_json_path, output_process_img_poly_path, output_csv):
    image_json = create_prediction(img_path, json_path, building_model_path, predictions_json_path, output_process_img_poly_path, output_csv)

    coords = image_json['features']['xy']
    wkt_polygons = [(coord['properties']['subtype'], coord['wkt']) for coord in coords]
    polygons = [(damage, shapely.wkt.loads(swkt)) for damage, swkt in wkt_polygons]

    image = blank_image()
    draw = ImageDraw.Draw(image, 'RGBA')

    for damage, polygon in polygons:
        x,y = polygon.exterior.coords.xy
        draw.polygon(list(zip(x,y)), mask_colors()[damage])

    return image

def original_mask(json_path):
    image_json = json.load(open(json_path))

    coords = image_json['features']['xy']
    wkt_polygons = [(0, coord['wkt']) for coord in coords]
    polygons = [(damage, shapely.wkt.loads(swkt)) for damage, swkt in wkt_polygons]

    image = blank_image()
    draw = ImageDraw.Draw(image, 'RGBA')

    for damage, polygon in polygons:
        x,y = polygon.exterior.coords.xy
        draw.polygon(list(zip(x,y)), mask_colors()[damage])

    return image

def colors():
    colors = {
        0: (0, 255, 0, 50),
        1: (255, 0, 0, 50),
    }
    return colors

def main_building(img_path, json_path, building_model_path, save_map_path, predictions_json_path, output_process_img_poly_path, output_csv, original_mask_path, predicted_mask_path):
    json_pred = create_prediction(img_path, json_path, building_model_path, predictions_json_path, output_process_img_poly_path, output_csv)
    coords = json_pred['features']['xy']
    wkt_polygons = [(coord['properties']['subtype'], coord['wkt']) for coord in coords]
    polygons = [(damage, shapely.wkt.loads(swkt)) for damage, swkt in wkt_polygons]

    image = Image.open(img_path)
    draw = ImageDraw.Draw(image, 'RGBA')

    for damage, polygon in polygons:
        x,y = polygon.exterior.coords.xy
        draw.polygon(list(zip(x,y)), colors()[damage])
    image.save(save_map_path, 'png')
    
    original_mask_png = original_mask(json_path)
    original_mask_png.save(original_mask_path, 'png')

    predicted_mask_png = predicted_mask(img_path, json_path, building_model_path, predictions_json_path, output_process_img_poly_path, output_csv,)
    predicted_mask_png.save(predicted_mask_path, 'png')

    return image

def predict_building(target_area, src):
    # import files
    dest = '/content/main/images'
    if not os.path.exists(dest + '/' + target_area):
        shutil.copytree(src + '/' + target_area, dest + '/' + target_area)
    

    src_model = src + '/Model'
    dest = '/content/main/models'
    if not os.path.exists(dest):
        shutil.copytree(src_model, dest)
    
    # create img and json path
    post_img = ''
    pre_img = ''
    json_img = ''
    for f in os.listdir('/content/main/images' + '/' + target_area):
        if f.endswith('post_disaster.png'):
            post_img = post_img + f
        elif f.endswith('pre_disaster.png'):
            pre_img = pre_img + f
        else:
            json_img = json_img + f

    images_home_path = '/content/main/images'

    img_path = images_home_path + '/' + target_area + '/' + post_img
    json_path = images_home_path + '/' + target_area + '/' + json_img

    # create model path
    models_home_path = '/content/main/models'
    building_model_path = models_home_path + '/' + 'Building Damage Detection Model.hdf5'
    road_model_path = models_home_path + '/' + 'Road Extraction Model.hdf5'

    # create prediction path
    output_home_path = '/content/prediction'
    os.makedirs(output_home_path + '/' + target_area, exist_ok = True)
    predictions_json_path = output_home_path + '/' + target_area + '/' + json_img
    output_process_img_poly_path = output_home_path + '/' + target_area + '/output_process_img_poly'
    os.makedirs(output_process_img_poly_path, exist_ok = True)
    output_csv = output_home_path + '/' + target_area + '/output_process_img_poly.csv'

    # create map output path
    save_map_home_path = '/content/mapping'
    os.makedirs(save_map_home_path, exist_ok = True)
    save_map_path = save_map_home_path + '/' + target_area + ' After' + '.jpg'

    # copy pre disaster satellite image
    pre_img_path = images_home_path + '/' + target_area + '/' + pre_img
    
    shutil.copy(pre_img_path, save_map_home_path + '/' + target_area + ' Before' + '.jpg')

    # create road path
    temporary_home_path = '/content/temporary'
    os.makedirs(temporary_home_path + '/' + target_area, exist_ok = True)
    predicted_mask_path = temporary_home_path + '/' + target_area + '/' + 'predicted_mask.png'
    original_mask_path = temporary_home_path + '/' + target_area + '/' + 'original_mask.png'

    

    return (img_path, json_path, building_model_path, save_map_path, predictions_json_path, output_process_img_poly_path, output_csv, original_mask_path, predicted_mask_path)

def run_all_inference(src):
    target_list = []
    for file in os.listdir(src):
        if file.startswith('Map'):
            target_list.append(file)
  
    for target_area in target_list:
        main_building(*predict_building(target_area, src))