# Deteccion-de-EPPs-YOLOv8

Pasos para utilizar la version entrenada por nosotros son los siguientes.

1) Crear una carpeta para contener todos los archivos.
2) Descargar el archivo "best.pt".
3) Instalar la libreria ultralytics, numpy 1.26.4 y cv2.
4) Descargar y abrir el codigo del archivo "test.py".
5) Modificar la ruta en la linea 5 con la ruta del archivo descargado "best.pt".
6) Modificar la ruta en la linea 8 con la ruta del archivo a anlizar. Puede ser imagen o video.
7) Ejecutar el codigo, el resultado quedará en la misma carpeta con el nombre "results_nombreimagen".

NOTA: En caso de contener un error al ejecutar el codigo, puede ser necesario ocupar el siguiente comando en el terminal "pip install --upgrade ultralytics" o "pip3install --upgrade ultralytics".

Metricas de detección.

- Precision Media:  0.7723
- Recall Medio:     0.6534
- F1-Score Medio:   0.6617

Métricas oficiales de YOLOv8.

- mAP50:  0.7341
- mAP95:  0.5062
