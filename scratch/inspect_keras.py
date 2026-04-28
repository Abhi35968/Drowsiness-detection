import tensorflow as tf

try:
    print("Checking drowsiness_model.keras...")
    model = tf.keras.models.load_model("drowsiness_model.keras", compile=False)
    model.summary()
    
    print("\nFirst 10 layers:")
    for i, layer in enumerate(model.layers[:10]):
        print(f"{i}: {layer.name} ({layer.__class__.__name__})")
        
except Exception as e:
    print(f"Error: {e}")
