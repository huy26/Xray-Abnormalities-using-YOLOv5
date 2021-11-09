from flask import Flask, render_template, request, Markup
import os
import random
from numpy import random

import torch
import cv2
from utils.general import check_img_size

from utils.datasets import LoadImages
from utils.general import (non_max_suppression,scale_coords, set_logging)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from models.experimental import attempt_load
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = "static"

# Parameter initialization
weights = 'models/best.pt'
set_logging()
device = select_device('')
half = device.type != 'cpu'
imgsize=640

# Load model
model = attempt_load(weights, map_location=device)
stride=int(model.stride.max())
imgsize= check_img_size(imgsize, stride)

if half:
    model.half()
    
# Request
@app.route("/", methods = ['POST', 'GET'])
def home_page():
    if request.method == 'GET':
        return render_template("index.html")
    else:
        image = request.files['file']
        if image:
            print(image.filename)
            print(app.config['UPLOAD_FOLDER'])
            source = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            print("Save = ", source)
            image.save(source)
            
            save_img = True
            dataset = LoadImages(source, img_size = imgsize, stride=stride)
            
            names = model.module.names if hasattr(model, 'module') else model.names
            
            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1,3,imgsize,imgsize).to(device).type_as(next(model.parameters()))) # run once
            
            conf_thres = 0.25
            iou_thres = 0.25
            
            for path, img, im0s, vid_cap, s in dataset:
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float() # convert image from uint8 to float16/32
                img /= 255.0
                if len(img.shape) ==3:
                    img = img.unsqueeze(0)

                # Inference
                pred = model(img, augment = False)[0]
                
                # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes = None, agnostic=False)
                
                extra = ""
                
                # Process detections
                for i, det in enumerate(pred):
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                    save_path = source
                    annotator = Annotator(im0, line_width=3, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_img: # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            annotator.box_label(xyxy, label, color=colors(int(cls), True))
                            extra += "<br>- <b>" + str(names[int(cls)]) + "</b> with confidence <b>{:.2f}% </b>".format(conf)
                
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                
        return render_template("index.html", user_image = image.filename, rand=random.random(), msg ="Upload file successfully", extra=Markup(extra))
                        
# Start server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)