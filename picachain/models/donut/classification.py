import re
from typing import List, Union

from PIL import Image


class DonutDocClassification:
    model_id = "naver-clova-ix/donut-base-finetuned-rvlcdip"

    def __init__(self) -> None:
        try:
            import torch
            from transformers import DonutProcessor, VisionEncoderDecoderModel
        except ImportError:
            raise ImportError(
                "Failed to import transformers. Install transformers using `pip install transformers torch sentencepiece`."
            )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = DonutProcessor.from_pretrained(self.model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_id).to(
            self.device
        )

    def classify(self, image: Union[str, Image.Image], *args, **kwargs) -> dict:
        """Extract information and answer the questions from the image.

        Args:
        - image (str | Image): can be full path or pillow.Image object.
        - queries (list): list of questions to be answered.

        Returns:
        - list of answers.
        """
        query_img = None

        if isinstance(image, str):
            query_img = Image.open(image)
        elif isinstance(image, Image.Image):
            query_img = image

        task_prompt = ""

        decoder_input_ids = self.processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        )["input_ids"]
        pixel_values = self.processor(query_img, return_tensors="pt").pixel_values

        outputs = self.model.generate(
            pixel_values.to(self.device),
            decoder_input_ids=decoder_input_ids.to(self.device),
            max_length=self.model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_scores=True,
        )
        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(
            self.processor.tokenizer.pad_token, ""
        )
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
        result = self.processor.token2json(sequence)
        if result:
            return result
        else:
            return {}
