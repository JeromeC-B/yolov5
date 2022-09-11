"""
But du script: c'est de prendre le fichier annotations.csv et créer le directory labels pour diffents modèles
"""

import datetime
import os
import sys
import time
import yaml
import ast


import detect

if __name__ == '__main__':

    print('début: ', os.path.abspath(__file__))
    print("heure: ", datetime.datetime.now())
    print('\n')

    time_debut = time.time()

    with open(r"C:\projets\yolov5\src_jer\airbus_aircraft\configs\yolov5\detect_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    detect.run(weights=config["weights"],
               source=config["source"],
               data=config["data"],
               imgsz=ast.literal_eval(config["imgsz"]),
               device=config["device"],
               project=config["project"])

    # detect.run(weights=r"C:\projets\yolov5\yolov5s.pt", source=r"C:\projets\external\database\airbus-aircraft-detection\raw-data\archive\extras",
    #            data=r"C:\projets\airbus-aircraft-detection\configs\dataset.yaml", imgsz=(2560, 2560), device=0,
    #            project=r"C:\projets\external\database\airbus-aircraft-detection\data-2022-09-10")

    print('\n')
    print("temps d'exécution: ", datetime.timedelta(seconds=time.time() - time_debut))
    print("heure: ", datetime.datetime.now())
    print('fin')

