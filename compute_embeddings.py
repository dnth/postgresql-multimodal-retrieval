import os

import numpy as np
import torch
from datasets import load_dataset
from loguru import logger
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast


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


def compute_embeddings(images, processor, model, device, batch_size=128):
    logger.info("Computing image embeddings")
    image_arr = None

    logger.info(f"Processing images in batches of {batch_size}")
    for i in tqdm(range(0, len(images), batch_size)):
        batch = images[i : i + batch_size]
        batch = processor(text=None, images=batch, return_tensors="pt", padding=True)[
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
    logger.info("Loading dataset")
    dataset = load_dataset("UCSC-VLAA/Recap-COCO-30K")

    save_dir = "./saved_images_coco_30k/"
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving images to {save_dir}")

    # Only process the first 100 images for testing and debugging
    dataset["train"] = dataset["train"].select(range(100))
    dataset = dataset.map(lambda x: save_image_to_disk(x, save_dir))

    logger.info("Extracting images from dataset")
    images = [dataset["train"][i]["image"] for i in range(len(dataset["train"]))]
    logger.info(f"Total images: {len(images)}")

    device, tokenizer, processor, model = initialize_model()

    image_arr = compute_embeddings(images, processor, model, device)

    # Normalize embeddings
    image_arr = image_arr.T / np.linalg.norm(image_arr, axis=1)

    # Save embeddings to a .npy file
    embeddings_file = "image_embeddings.npy"
    np.save(embeddings_file, image_arr)
    logger.info(f"Embeddings saved to {embeddings_file}")


if __name__ == "__main__":
    logger.info("Starting main process")
    main()
    logger.info("Process completed")
