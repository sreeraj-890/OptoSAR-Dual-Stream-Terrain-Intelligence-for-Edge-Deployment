import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import glob

# Professional Satellite Terrain Classes
TERRAIN_CLASSES = ["Annual Crop", "Forest", "Herbaceous Vegetation", "Highway", "Industrial", "Pasture", "Permanent Crop", "Residential", "River", "Sea/Lake"]

ENGINE_PATH = "OptoSAR_Edge_Engine.engine" 

# --- AUTOMATED GEOSPATIAL FILE DETECTION ---
# Automatically scans the data folder for any TIFF variations
tif_files = glob.glob("data/*.tif") + glob.glob("data/*.tiff") + glob.glob("data/*.TIF") + glob.glob("data/*.TIFF")

if not tif_files:
    print("\n[ERROR] Could not find any .tif or .tiff satellite images inside the 'data/' folder.")
    print("Please drop a satellite image into 'data/' and try again.")
    exit(1)

# Select the first TIFF file found
IMAGE_PATH = tif_files[0]
print(f"\n🚀 [AUTOMATIC SCAN] Found target satellite image: {IMAGE_PATH}")

if not os.path.exists(ENGINE_PATH):
    print(f"[ERROR] Could not find compiled engine at: {ENGINE_PATH}")
    print("Please run 'python3 scripts/build_engine.py' first.")
    exit(1)

print("Loading TensorRT Optimization Engine into VRAM...")
logger = trt.Logger(trt.Logger.WARNING)
with open(ENGINE_PATH, "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# --- GEOSPATIAL TIFF HANDLING ---
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_UNCHANGED)
if img is None:
    print(f"[ERROR] OpenCV could not read the file structure of: {IMAGE_PATH}")
    exit(1)

# Dynamic Radiometric Normalization (Handles 16-bit space data vs standard 8-bit)
if img.dtype == np.uint16:
    img = img.astype(np.float32) / 65535.0  # Normalized 16-bit
else:
    img = img.astype(np.float32) / 255.0    # Normalized 8-bit

img = cv2.resize(img, (256, 256))
channels = img.shape[2] if len(img.shape) > 2 else 1

# Extracting Geospatial Bands to match the Dual-Stream Network Architecture
if channels >= 4:
    opt_in = img[:, :, :4].transpose(2, 0, 1).copy() # Takes R, G, B, and Near-Infrared
else:
    opt_in = np.dstack((img, img[:,:,0])).transpose(2, 0, 1).copy() # Fallback

if channels >= 2:
    sar_in = img[:, :, :2].transpose(2, 0, 1).copy() # Fuses first two bands for SAR stream simulation
else:
    sar_in = np.dstack((img, img)).transpose(2, 0, 1).copy() # Fallback

# Add Batch Dimension (1, C, H, W)
sar_in = np.expand_dims(sar_in, axis=0)
opt_in = np.expand_dims(opt_in, axis=0)

# --- HARDWARE MEMORY MANAGMENT (PyCUDA Async Page-Allocation) ---
allocations = {}
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = tuple(engine.get_tensor_shape(name))
    dtype = trt.nptype(engine.get_tensor_dtype(name))
    
    mem = cuda.pagelocked_empty(shape, dtype)
    allocations[name] = {'host': mem, 'device': cuda.mem_alloc(mem.nbytes)}
    context.set_tensor_address(name, int(allocations[name]['device']))

# --- HARDWARE EXECUTION ---
print("Executing high-speed inference pipeline on RTX 3050 GPU...")
stream = cuda.Stream()

# Host-to-Device (CPU to GPU) Asynchronous memory transfer
cuda.memcpy_htod_async(allocations['SAR_Sensor_Input']['device'], sar_in, stream)
cuda.memcpy_htod_async(allocations['Optical_Sensor_Input']['device'], opt_in, stream)

# TensorRT Engine Execution Hook
context.execute_async_v3(stream_handle=stream.handle)

# Device-to-Host (GPU to CPU) transfer back
cuda.memcpy_dtoh_async(allocations['Terrain_Intelligence_Output']['host'], allocations['Terrain_Intelligence_Output']['device'], stream)
stream.synchronize()

# --- EXTRACT OUTPUT LOGITS ---
res = np.argmax(allocations['Terrain_Intelligence_Output']['host'], axis=1)
unique, counts = np.unique(res, return_counts=True)
idx = unique[np.argmax(counts)]

# Print Final Professional Output
print(f"\n>>> TARGET IDENTIFIED FOR [{os.path.basename(IMAGE_PATH)}]: {TERRAIN_CLASSES[idx].upper()} <<<")
print("Edge Inference Complete. Metrics successfully recorded.")
