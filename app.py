import os
import streamlit as st
from PIL import Image

from main import *


st.markdown(
    """
    # Xview2 Building Damage Detection
    """
)

class application():
    
    def __init__(self):
        
        if "img_poly_path" not in st.session_state:
            st.session_state["img_poly_path"] = {}

        if "save_map_path" not in st.session_state:
            st.session_state["save_map_path"] = {}
        
        if "output_csv" not in st.session_state:
            st.session_state["output_csv"] = ""
        
        if "model_path" not in st.session_state:
            st.session_state["model_path"] = {}
        
        if "post_img" not in st.session_state:
            st.session_state["post_img"] = {}

        if "json_img" not in st.session_state:
            st.session_state["json_img"] = {}

        if "pre_image" not in st.session_state:
            st.session_state["pre_image"] = {}
            
        if "post_image" not in st.session_state:
            st.session_state["post_image"] = {}
        
        with st.form("Choose Location"):
            choosen_images = st.selectbox(
                'Choose Location',
                ('north-tondo-palu', 'lere-palu'))

            self.submitted = st.form_submit_button("Preview Map")
            
            if self.submitted:

                
                target_area = "data/" + choosen_images
                result_path = "result"
                st.session_state["img_poly_path"] = result_path + "/img_poly"
                st.session_state["save_map_path"] =result_path + "/result.jpg"
                st.session_state["output_csv"] = result_path + "/result.csv"
                st.session_state["model_path"] = "model-building.hdf5"
                os.makedirs(st.session_state["img_poly_path"], exist_ok = True)

                for f in os.listdir(target_area):
                    if f.endswith('post_disaster.png'):
                        st.session_state["post_img"] = target_area + "/" + f
                    elif f.endswith('pre_disaster.png'):
                        pre_img = target_area + "/" + f
                    else:
                        st.session_state["json_img"] = target_area + "/" + f
                st.session_state["pre_image"] = Image.open(pre_img)
                st.session_state["post_image"] = Image.open(st.session_state["post_img"])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(st.session_state["pre_image"], caption="Pre Disaster")
                
                with col2:
                    st.image(st.session_state["post_image"], caption="Pasca Disaster")
                    
        with st.form("Detect Damaged Building"):
            
            self.submitted = st.form_submit_button("Analyze")
            
            if self.submitted:
                json_pred = create_prediction(
                    st.session_state["post_img"],
                    st.session_state["json_img"], 
                    st.session_state["model_path"],
                    st.session_state["json_img"], 
                    st.session_state["img_poly_path"], 
                    st.session_state["output_csv"])

                coords = json_pred['features']['xy']
                wkt_polygons = [(coord['properties']['subtype'], coord['wkt']) for coord in coords]
                polygons = [(damage, shapely.wkt.loads(swkt)) for damage, swkt in wkt_polygons]

                image = Image.open(st.session_state["post_img"])
                draw = ImageDraw.Draw(image, 'RGBA')

                for damage, polygon in polygons:
                    x,y = polygon.exterior.coords.xy
                    draw.polygon(list(zip(x,y)), colors()[damage])
                image.save(st.session_state["save_map_path"], 'png')
                col1, col2 = st.columns(2)
                with col1:
                    st.image(st.session_state["pre_image"], caption="Pre Disaster")
                
                with col2:
                    st.image(st.session_state["post_image"], caption="Pasca Disaster")
                
                st.markdown("### Result")
                st.image("result/result.jpg", caption="Pasca Disaster")
            
    
    def window(self):
        st.write(" ")
        
app = application()
app.window()