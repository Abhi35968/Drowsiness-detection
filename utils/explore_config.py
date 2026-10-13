import json, zipfile

with zipfile.ZipFile("drowsiness_model.keras") as z:
    config = json.loads(z.read("config.json"))

outer_layers = config["config"]["layers"]
print("=== OUTER MODEL LAYERS (in order) ===")
for layer in outer_layers:
    name = layer.get("name", "?")
    cls = layer.get("class_name", "?")
    print(f"  {name} ({cls})")
    
    if cls == "Functional":
        inner_layers = layer.get("config", {}).get("layers", [])
        print(f"    -> {len(inner_layers)} internal layers")
        for il in inner_layers[:8]:
            iname = il.get("name", "?")
            icls = il.get("class_name", "?")
            print(f"       {iname} ({icls})")
        print("       ...")
