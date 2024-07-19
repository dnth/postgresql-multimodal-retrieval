import numpy as np
import torch
from loguru import logger
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast


class CLIP:
    def __init__(
        self, model_id: str = "openai/clip-vit-base-patch32", device: str = None
    ) -> None:
        logger.info(f"Initializing CLIP model: {model_id}")
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        logger.info(f"Using device: {self.device}")

        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)

    def encode_image(self, image_paths: str, batch_size: int = 128) -> np.ndarray:
        logger.info(f"Computing image embeddings in batches of {batch_size}")
        image_embeddings = None

        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i : i + batch_size]

            # Load and process images
            batch_images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    batch_images.append(img)
                except Exception as e:
                    logger.error(f"Error loading image {path}: {str(e)}")
                    continue

            if not batch_images:
                logger.warning(f"No valid images in batch starting at index {i}")
                continue

            batch = self.processor(
                text=None, images=batch_images, return_tensors="pt", padding=True
            )["pixel_values"].to(self.device)

            batch_emb = self.model.get_image_features(pixel_values=batch)
            batch_emb = batch_emb.squeeze(0)
            batch_emb = batch_emb.cpu().detach().numpy()

            if image_embeddings is None:
                image_embeddings = batch_emb
            else:
                image_embeddings = np.concatenate((image_embeddings, batch_emb), axis=0)

        image_embeddings = image_embeddings.T / np.linalg.norm(image_embeddings, axis=1)

        logger.info(
            f"Finished processing. Final embedding shape: {image_embeddings.shape}"
        )
        return image_embeddings

    def encode_text(self, text: str) -> np.ndarray:
        logger.info(f"Computing text embedding for: {text}")
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        text_emb = self.model.get_text_features(**inputs)
        text_emb = text_emb.cpu().detach().numpy()
        return text_emb.flatten()
