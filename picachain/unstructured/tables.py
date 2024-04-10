from pathlib import PurePath
from typing import Any, Union

from PIL import Image

from picachain.pydantic import BaseModel

Path = Union[str, PurePath]


class Table(BaseModel):
    content: Any
    metadata: dict


class TableParser:
    @classmethod
    def from_image(cls, image: Union[Path, Image.Image]):
        if isinstance(image, Path):
            image = Image.open(image)
