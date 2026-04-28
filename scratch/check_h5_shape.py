import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D

class PatchedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(**kwargs)

try:
    model = tf.keras.models.load_model(
        "drowsiness_model.h5", 
        custom_objects={'DepthwiseConv2D': PatchedDepthwiseConv2D},
        compile=False
    )
    print(f"Input shape: {model.input_shape}")
except Exception as e:
    print(f"Error: {e}")
