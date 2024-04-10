from PIL import Image


class Deplot:
    model_id = "google/deplot"

    def __init__(self) -> None:
        try:
            from transformers import (
                Pix2StructForConditionalGeneration,
                Pix2StructProcessor,
            )
        except ImportError:
            raise ImportError(
                "Failed to import transformers. Install using `pip install transformers`"
            )

        self.processor = Pix2StructProcessor.from_pretrained(self.model_id)
        self.model = Pix2StructForConditionalGeneration.from_pretrained(self.model_id)

    def generate(self, image: Image.Image):
        inputs = self.processor(
            images=image,
            text="Generate underlying data table of the figure below:",
            return_tensors="pt",
        )
        predictions = self.model.generate(**inputs, max_new_tokens=512)
        result = self.processor.decode(predictions[0], skip_special_tokens=True)
        if result is not None:
            return result
        else:
            return ""
