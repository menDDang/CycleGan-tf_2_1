from PIL import Image
import numpy as np

MAX_RGB_VALUE = 255.0

def read_image(input_file_name, apply_normalization=True):
    x = np.asarray(Image.open(input_file_name).convert('RGB'), dtype=np.float32)
    if apply_normalization:
        x /= MAX_RGB_VALUE 
        
    return x

def write_image(x, output_file_name, isNormalized=True):
    x *= MAX_RGB_VALUE
    x = Image.fromarray(x.astype(np.uint8))
    x.save(output_file_name)
