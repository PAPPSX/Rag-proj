import io
import os
from dotenv import load_dotenv
import yaml
from PIL import Image

load_dotenv()

def image_to_int_array(image, format="JPEG"):
    """Current Workers AI REST API consumes an array of unsigned 8 bit integers"""
    bytes = io.BytesIO()
    image.save(bytes, format=format)
    return list(bytes.getvalue())

