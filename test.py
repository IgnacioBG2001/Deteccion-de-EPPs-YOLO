from ultralytics import YOLO
import os
import cv2

# ================= CONFIGURACIÓN =================
# Ruta al modelo
model = YOLO(r"c:")

# Ruta del archivo de entrada
path = r"c:"

# Ruta de salida (donde se guardará el video final)
output_path = r"c:"

CONF_MIN = 0.30  # Confianza mínima para mantener una detección

# ================= FUNCIONES =================

def filtrar_confianza(result):
    """
    Filtra las cajas del resultado que tengan una confianza menor a CONF_MIN.
    Modifica el objeto result in-situ.
    """
    if result.boxes is None:
        return

    keep = []
    for i, box in enumerate(result.boxes):
        conf = float(box.conf[0])
        if conf >= CONF_MIN:
            keep.append(i)
    
    # Sobrescribimos las boxes con solo las que pasaron el filtro
    if keep:
        result.boxes = result.boxes[keep]
    else:
        # Si ninguna cumple, dejamos las boxes vacías
        result.boxes = result.boxes[[]] 

# ================= LÓGICA PRINCIPAL =================

image_exts = [".jpg", ".jpeg", ".png", ".bmp"]
video_exts = [".mp4", ".avi", ".mov", ".mkv"]

ext = os.path.splitext(path)[1].lower()

# 1. PROCESAMIENTO DE IMAGEN
if ext in image_exts:
    print(f"Procesando imagen: {path}")
    results = model(path)
    
    for r in results:
        filtrar_confianza(r)
        
        # Guardar imagen procesada
        output_img_path = path.replace(ext, f"_out{ext}")
        r.save(filename=output_img_path)
        r.show() # Opcional: mostrar en pantalla
        print(f"Imagen guardada en: {output_img_path}")


# 2. PROCESAMIENTO DE VIDEO
elif ext in video_exts:
    print(f"Procesando video: {path}")
    
    # Capturamos el video original
    cap = cv2.VideoCapture(path)
    
    # Obtenemos propiedades del video original para crear el nuevo igual
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Definimos el codec y el objeto VideoWriter
    # 'mp4v' es un codec estándar para .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # Fin del video
        
        # Procesamos el frame con YOLO
        # persist=True ayuda a mantener IDs si usas tracking, aunque no es estricto aquí
        # verbose=False para que no llene la consola de texto
        results = model(frame, verbose=False) 
        
        for r in results:
            # 1. Filtramos las cajas débiles
            filtrar_confianza(r)
            
            # 2. "Plot" dibuja las cajas en el frame y devuelve una imagen (array numpy)
            # Esto es lo que queremos guardar en el video
            frame_procesado = r.plot()
            
            # 3. Escribimos el frame pintado en el archivo de video
            out.write(frame_procesado)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Procesando frame {frame_count}...")

    # Liberamos recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print("========================================")
    print(f"Video finalizado y guardado en: {output_path}")
    print("========================================")

else:
    print("Extensión no soportada:", ext)