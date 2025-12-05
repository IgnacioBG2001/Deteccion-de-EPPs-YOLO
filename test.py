from ultralytics import YOLO
import os

# Ruta a tu modelo
model = YOLO(r"c:\Users\bravo\Desktop\P3\best.pt")

# Ruta al archivo (imagen o video)
path = r"c:\Users\bravo\Desktop\P3\hola.jpg"  # Cambia a imagen o video

# Detectar si es imagen o video según la extensión
ext = os.path.splitext(path)[1].lower()
image_exts = [".jpg", ".jpeg", ".png", ".bmp"]
video_exts = [".mp4", ".avi", ".mov", ".mkv"]

# Procesamiento
if ext in image_exts:
    print("Procesando imagen...")
    results = model(path, show=True, save=True)

    for r in results:
        print(r.boxes)

elif ext in video_exts:
    print("Procesando video...")
    # save=True guarda el video anotado en runs/detect/predict
    results = model(path, show=True, save=True)

    print("Detección en video finalizada.")
else:
    print("❌ Extensión no soportada:", ext)
