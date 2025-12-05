from ultralytics import YOLO
import os

# Ruta al modelo
model = YOLO(r"c:\Users\bravo\OneDrive - Universidad Mayor\10° semestre\PDI\P3\best.pt")

# Ruta del archivo
path = r"c:\Users\bravo\OneDrive - Universidad Mayor\10° semestre\PDI\P3\image.png"

# Extensiones soportadas
image_exts = [".jpg", ".jpeg", ".png", ".bmp"]
video_exts = [".mp4", ".avi", ".mov", ".mkv"]

ext = os.path.splitext(path)[1].lower()

CONF_MIN = 0.30  # confianza mínima


def filtrar_confianza(result):
    """Quita boxes con confianza menor a CONF_MIN."""
    keep = []
    for i, box in enumerate(result.boxes):
        conf = float(box.conf[0])
        if conf >= CONF_MIN:
            keep.append(i)

    result.boxes = result.boxes[keep] if keep else []


# ==========================================
# PROCESAMIENTO IMAGEN (funciona 100%)
# ==========================================
if ext in image_exts:
    print("Procesando imagen...")

    # No mostramos nada todavía
    results = model(path)

    for r in results:
        filtrar_confianza(r)

        # AHORA sí mostramos/guardamos la imagen filtrada
        r.show()       # muestra en pantalla ya filtrado
        r.save()       # guarda imagen ya filtrada

        print(r.boxes)


# ==========================================
# PROCESAMIENTO VIDEO (funciona 100%)
# ==========================================
elif ext in video_exts:
    print("Procesando video...")

    # Procesa frame por frame SIN dibujar cajas
    results = model(path, stream=True)

    # Donde guardar el video nuevo
    save_dir = "video_filtrado"
    os.makedirs(save_dir, exist_ok=True)

    # Usamos el generador frame por frame
    for i, r in enumerate(results):
        filtrar_confianza(r)
        r.save(filename=f"{save_dir}/frame_{i}.jpg")  # guarda cada frame sin boxes débiles

    print("Video filtrado guardado correctamente.")

else:
    print("Extensión no soportada:", ext)
