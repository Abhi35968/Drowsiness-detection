"""
DEFINITIVE Keras 3 -> Keras 2 model converter.
Maps weights by layer TYPE + creation order, which is the only correct approach
when many layers share identical weight shapes (e.g., BatchNormalization).
"""
import json, zipfile, h5py, numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, Rescaling
from tensorflow.keras.models import Model
from collections import defaultdict

IMG_SIZE = 96

# ==============================
# Step 1: Read config.json for layer ordering + types
# ==============================
print("Reading model config...")
with zipfile.ZipFile("drowsiness_model.keras") as z:
    config = json.loads(z.read("config.json"))
    z.extract("model.weights.h5", ".")

outer_layers = config["config"]["layers"]

# Get MobileNetV2 internal layers in topological order (from config)
mobilenet_config_layers = []
for layer in outer_layers:
    if layer.get("class_name") == "Functional":
        for il in layer["config"]["layers"]:
            mobilenet_config_layers.append({
                "name": il["name"],
                "class_name": il["class_name"]
            })

# Keras 3 naming convention: each layer type gets sequential numbers
# conv2d, conv2d_1, conv2d_2, ... (in creation/topological order)
# batch_normalization, batch_normalization_1, ...
# depthwise_conv2d, depthwise_conv2d_1, ...

# Map Keras class names to their Keras 3 base prefix
CLASS_TO_K3_PREFIX = {
    "Conv2D": "conv2d",
    "BatchNormalization": "batch_normalization",
    "DepthwiseConv2D": "depthwise_conv2d",
}

# Only these layer types have weights
WEIGHT_LAYER_TYPES = {"Conv2D", "BatchNormalization", "DepthwiseConv2D"}

# Build the mapping: config_name (K2) -> k3_name (H5 file)
type_counters = defaultdict(int)
config_to_k3 = {}

for layer_info in mobilenet_config_layers:
    cls = layer_info["class_name"]
    config_name = layer_info["name"]
    
    if cls in WEIGHT_LAYER_TYPES:
        prefix = CLASS_TO_K3_PREFIX[cls]
        count = type_counters[cls]
        
        if count == 0:
            k3_name = prefix  # First one has no suffix
        else:
            k3_name = f"{prefix}_{count}"
        
        type_counters[cls] += 1
        config_to_k3[config_name] = k3_name

print(f"Built mapping for {len(config_to_k3)} MobileNetV2 layers")
# Show first few mappings
for k, v in list(config_to_k3.items())[:8]:
    print(f"  {k:35s} -> {v}")
print("  ...")

# ==============================
# Step 2: Build the Keras 2 model
# ==============================
print("\nBuilding model in TF 2.15...")
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights=None)
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = Rescaling(scale=1./127.5, offset=-1.0)(inputs)
x = base_model(x)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs, outputs)

# ==============================
# Step 3: Load weights using the correct mapping
# ==============================
print("\nLoading weights with correct mapping...")
h5 = h5py.File("model.weights.h5", "r")

def read_layer_weights(h5_group, layer_name):
    if layer_name not in h5_group:
        return None
    vars_g = h5_group[layer_name].get("vars", {})
    if not vars_g:
        return None
    indices = sorted([int(k) for k in vars_g.keys()])
    return [np.array(vars_g[str(i)]) for i in indices]

h5_mobilenet = h5["layers"]["functional"]["layers"]
h5_top = h5["layers"]

# Build K2 layer lookup
k2_layer_map = {l.name: l for l in base_model.layers}

loaded = 0
errors = 0

for config_name, k3_name in config_to_k3.items():
    if config_name not in k2_layer_map:
        print(f"  SKIP: {config_name} not in K2 model")
        continue
    
    ws = read_layer_weights(h5_mobilenet, k3_name)
    if ws is None:
        print(f"  MISS: {k3_name} not found in H5")
        errors += 1
        continue
    
    k2_layer = k2_layer_map[config_name]
    k2_shapes = [tuple(w.shape) for w in k2_layer.get_weights()]
    h5_shapes = [tuple(w.shape) for w in ws]
    
    if k2_shapes != h5_shapes:
        print(f"  SHAPE MISMATCH: {config_name} ({k2_shapes}) vs {k3_name} ({h5_shapes})")
        errors += 1
        continue
    
    k2_layer.set_weights(ws)
    loaded += 1

# Load top-level layers (dense, dense_1)
for layer in model.layers:
    if layer.name in ("dense", "dense_1"):
        ws = read_layer_weights(h5_top, layer.name)
        if ws:
            layer.set_weights(ws)
            loaded += 1
            print(f"  Loaded top layer: {layer.name}")

h5.close()
print(f"\nLoaded {loaded} layers, {errors} errors")

# ==============================
# Step 4: Save
# ==============================
print("Saving as drowsiness_model.h5...")
model.save("drowsiness_model.h5")

import os
os.remove("model.weights.h5")
print("DONE! drowsiness_model.h5 is ready.")
