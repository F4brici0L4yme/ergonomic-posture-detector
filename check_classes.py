import yolov5
import torch

MODEL_PATH = "small640.pt"

# Detectar GPU o CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Cargar modelo sin cámara ni nada más
model = yolov5.load(MODEL_PATH, device=device)

# Imprimir clases
print("\n=== CLASES DEL MODELO ===")
for class_id, class_name in model.names.items():
    print(f"ID {class_id}: {class_name}")
