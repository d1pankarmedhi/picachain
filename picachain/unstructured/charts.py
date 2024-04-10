from pathlib import PurePath
from typing import Any, Union

from PIL import Image

from picachain.models import Deplot
from picachain.pydantic import BaseModel

Path = Union[str, PurePath]


class Chart(BaseModel):
    content: str


class ChartParser:
    @classmethod
    def from_image(cls, image: Union[Path, Image.Image]) -> Chart:
        if isinstance(image, Path):
            image = Image.open(image)

        deplot = Deplot()
        result = deplot.generate(image)

        return Chart(
            content=result,
        )
