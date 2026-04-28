import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D

class PatchedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(**kwargs)

try:
    print("Checking drowsiness_model.h5...")
    model = tf.keras.models.load_model(
        "drowsiness_model.h5", 
        custom_objects={'DepthwiseConv2D': PatchedDepthwiseConv2D},
        compile=False
    )
    model.summary()
    
    print("\nFirst 10 layers:")
    for i, layer in enumerate(model.layers[:10]):
        print(f"{i}: {layer.name} ({layer.__class__.__name__})")
        
except Exception as e:
    print(f"Error: {e}")
