import os

import numpy as np
import torch

from loguru import logger
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast

from src.database import PostgreSQLDatabase
from src.datasets import HuggingFaceDatasets

from PIL import Image

def save_image_to_disk(example, save_dir):
    filename = f"{example['image_id']}.jpg"
    filepath = os.path.join(save_dir, filename)
    example["image"].save(filepath)
    return {"image_filepath": filepath}


def initialize_model(model_id="openai/clip-vit-base-patch32"):
    logger.info(f"Initializing CLIP model: {model_id}")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    logger.info(f"Using device: {device}")

    tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)

    return device, tokenizer, processor, model


def compute_embeddings(image_paths, processor, model, device, batch_size=128):
    logger.info(f"Computing image embeddings in batches of {batch_size}")
    image_arr = None
    
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i : i + batch_size]
        
        # Load and process images
        batch_images = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                batch_images.append(img)
            except Exception as e:
                logger.error(f"Error loading image {path}: {str(e)}")
                continue

        if not batch_images:
            logger.warning(f"No valid images in batch starting at index {i}")
            continue

        batch = processor(text=None, images=batch_images, return_tensors="pt", padding=True)[
            "pixel_values"
        ].to(device)
        
        batch_emb = model.get_image_features(pixel_values=batch)
        batch_emb = batch_emb.squeeze(0)
        batch_emb = batch_emb.cpu().detach().numpy()
        
        if image_arr is None:
            image_arr = batch_emb
        else:
            image_arr = np.concatenate((image_arr, batch_emb), axis=0)

    logger.info(f"Finished processing. Final embedding shape: {image_arr.shape}")
    return image_arr


def main():

    ds = HuggingFaceDatasets("UCSC-VLAA/Recap-COCO-30K")
    ds.save_dataset_images('saved_images_coco_30k')
    image_filepaths = ds.get_image_paths()
    dataset_df = ds.to_pandas()

    device, tokenizer, processor, model = initialize_model()

    image_arr = compute_embeddings(image_filepaths, processor, model, device, batch_size=256)
    # image_arr = np.load("image_embeddings.npy")

    # Normalize embeddings
    image_arr = image_arr.T / np.linalg.norm(image_arr, axis=1)

    # Save embeddings to a .npy file
    embeddings_file = "image_embeddings.npy"
    np.save(embeddings_file, image_arr)
    logger.info(f"Embeddings saved to {embeddings_file}")

    db = PostgreSQLDatabase("retrieval_db")
    db.insert_data(dataset_df, image_arr)


if __name__ == "__main__":
    main()
