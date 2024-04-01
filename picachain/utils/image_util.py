import base64
import io

from PIL import Image


def image_to_base64(img: Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    img_byte_str = buffer.getvalue()
    return base64.b64encode(img_byte_str, altchars=b"-_").decode("utf-8")


def base64_to_image(base64_str: str) -> Image:
    img_data = base64.b64decode(base64_str, altchars=b"-_")
    img_buffer = io.BytesIO(img_data)
    img = Image.open(img_buffer)
    return img
