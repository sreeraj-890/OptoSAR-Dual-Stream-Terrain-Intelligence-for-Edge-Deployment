import tensorrt as trt
import os

# Initialize TensorRT builders
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

ONNX_PATH = "OptoSAR_Edge_Engine.onnx"

# 1. Verification Check
if not os.path.exists(ONNX_PATH):
    print(f"\n[ERROR] Could not find ONNX model at: {ONNX_PATH}")
    print("Please make sure your OptoSAR_Edge_Engine.onnx file is in this folder.")
    exit(1)

# 2. Parse the ONNX Model
print("Parsing ONNX file...")
with open(ONNX_PATH, "rb") as model:
    if not parser.parse(model.read()):
        print("ERROR: Failed to parse the ONNX file.")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit(1)

# 3. Configure Hardware Optimization Profiles
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)  # Enable high-speed FP16 precision
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # Allocate 1GB VRAM for optimization

# Match the dual-stream architecture shape
profile = builder.create_optimization_profile()
profile.set_shape("SAR_Sensor_Input", (1, 2, 256, 256), (1, 2, 256, 256), (1, 2, 256, 256))
profile.set_shape("Optical_Sensor_Input", (1, 4, 256, 256), (1, 4, 256, 256), (1, 4, 256, 256))
config.add_optimization_profile(profile)

# 4. Compile and Serialize the Engine
print("Building optimized TensorRT engine (This might take 1-3 minutes on your RTX 3050)...")
engine = builder.build_serialized_network(network, config)

if engine is None:
    print("\n[ERROR] FAILED to build the engine hardware definition.")
else:
    with open("OptoSAR_Edge_Engine.engine", "wb") as f:
        f.write(engine)
    print("\n>>> PASSED! OptoSAR_Edge_Engine.engine has been successfully generated and saved. <<<")
