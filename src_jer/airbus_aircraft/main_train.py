"""
But du script: c'est de prendre le fichier annotations.csv et créer le directory labels pour diffents modèles
"""

import datetime
import os
import sys
import time
import yaml
import ast


import train
from src_jer.airbus_aircraft.training import create_train_val_txt

if __name__ == '__main__':

    print('début: ', os.path.abspath(__file__))
    print("heure: ", datetime.datetime.now())
    print('\n')

    time_debut = time.time()

    with open(r"C:\projets\yolov5\src_jer\airbus_aircraft\configs\train_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    # ### faire les train/val .txt
    create_train_val_txt.create_train_val_txt_from_yaml(path_yaml=config["data"])

    train.run(weights=config["weights"],
              cfg=config["cfg"],
              data=config["data"],
              hyp=config["hyp"],
              epochs=config["epochs"],
              batch_size=config["batch_size"],
              imgsz=config["imgsz"],
              rect=config["rect"],
              resume=config["resume"],
              nosave=config["nosave"],
              noval=config["noval"],
              noautoanchor=config["noautoanchor"],
              noplots=config["noplots"],
              evolve=ast.literal_eval(config["evolve"]),
              bucket=config["bucket"],
              cache=ast.literal_eval(config["cache"]),
              image_weights=config["image_weights"],
              device=config["device"],
              multi_scale=config["multi_scale"],
              single_cls=config["single_cls"],
              optimizer=config["optimizer"],
              sync_bn=config["sync_bn"],
              workers=config["workers"],
              project=config["project"],
              name=config["name"],
              exist_ok=config["exist_ok"],
              quad=config["quad"],
              cos_lr=config["cos_lr"],
              label_smoothing=config["label_smoothing"],
              patience=config["patience"],
              freeze=config["freeze"],
              save_period=config["save_period"],
              seed=config["seed"],
              local_rank=config["local_rank"],
              entity=ast.literal_eval(config["entity"]),
              upload_dataset=config["upload_dataset"],
              bbox_interval=config["bbox_interval"],
              artifact_alias=config["artifact_alias"]
              )

    print('\n')
    print("temps d'exécution: ", datetime.timedelta(seconds=time.time() - time_debut))
    print("heure: ", datetime.datetime.now())
    print('fin')

