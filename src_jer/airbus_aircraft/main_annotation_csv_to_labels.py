"""
But du script: c'est de prendre le fichier annotations.csv et créer le directory labels pour diffents modèles
"""

import datetime
import os
import sys
import time

from src_jer.airbus_aircraft.annotations_csv_to_labels import yolov5

if __name__ == '__main__':
    print('début: ', os.path.abspath(__file__))
    print("heure: ", datetime.datetime.now())
    print('\n')

    time_debut = time.time()

    yolov5.annotations_to_labels(path_in=r"C:\projets\external\database\airbus-aircraft-detection\raw-data\archive\\",
                                 path_out=r"C:\projets\external\database\airbus-aircraft-detection\raw-data\archive\labels\\")

    print('\n')
    print("temps d'exécution: ", datetime.timedelta(seconds=time.time() - time_debut))
    print("heure: ", datetime.datetime.now())
    print('fin')

