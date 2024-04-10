from PIL.Image import Image


class Text2ImageSDXLTurbo:
    def __init__(
        self,
    ):
        try:
            import torch
            from diffusers import AutoPipelineForText2Image
        except ImportError as e:
            raise ImportError(
                "Could not import torch, diffusers."
                "Please install it with `pip install torch diffusers`"
            ) from e
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")

    def from_text(
        self,
        prompt: str,
        guidance_scale: float = 0.1,
        num_inference_steps: int = 50,
    ) -> Image:
        return self.pipeline(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).images[0]
