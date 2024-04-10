from collections import defaultdict
from typing import List, Union

from PIL import Image


class LLava:
    def __init__(self) -> None:
        try:
            import bitsandbytes
            import torch
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                LlavaForConditionalGeneration,
            )
        except ImportError:
            raise ImportError(
                "Failed to import transformers and bitsandbytes. Install using `pip install transformers torch accelerate bitsandbytes`."
            )

        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "llava-hf/llava-1.5-7b-hf"

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=self.quantization_config,
            device_map="auto",
        )

    @property
    def device(self):
        return self.device

    @property
    def model(self):
        return self.model_id

    @classmethod
    def config(cls, padding=True):
        pass

    def query(self, prompts: List[str], images: Union[List[str], List[Image.Image]]):
        if isinstance(images, List[str]):
            inference_images = [Image.open(img) for img in images]
        else:
            inference_images = images

        inputs = self.processor(
            prompts, inference_images, padding=True, return_tensors="pt"
        ).to(self.device)

        output = self.model.generate(**inputs, max_new_tokens=20)
        generated_text = self.processor.batch_decode(output, skip_special_tokens=True)

        result = defaultdict(str)
        for idx, text in enumerate(generated_text):
            result[prompts[idx]] = text.split("ASSISTANT:")[-1]

        return result
