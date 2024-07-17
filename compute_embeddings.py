import os

import numpy as np
import psycopg
import torch
from datasets import load_dataset
from loguru import logger
from pgvector.psycopg import register_vector
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast

from src.database import PostgreSQLDatabase

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


def insert_to_db(dataset, embeddings):
    df = dataset["train"].to_pandas()
    df = df.drop(columns=["image"])

    df["image_filepath"] = df["image_filepath"].apply(lambda x: x.split("/")[-1])

    df["img_emb"] = embeddings.T.tolist()

    conn = psycopg.connect(dbname="retrieval_db")
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    register_vector(conn)

    # Drop the existing table if it exists and create a new one
    cur.execute("""
    DROP TABLE IF EXISTS image_metadata;

    CREATE TABLE image_metadata (
        image_id INTEGER PRIMARY KEY,
        coco_url TEXT,
        caption TEXT,
        recaption TEXT,
        image_filepath TEXT,
        img_emb vector(512)
    )
    """)

    # Prepare the insert statement
    insert_sql = """
    INSERT INTO image_metadata (image_id, coco_url, caption, recaption, image_filepath, img_emb)
    VALUES (%s, %s, %s, %s, %s, %s)
    """

    # Convert dataframe to list of tuples
    data = []
    for _, row in df.iterrows():
        data.append(
            (
                row["image_id"],
                row["coco_url"],
                row["caption"],
                row["recaption"],
                row["image_filepath"],
                row["img_emb"],
            )
        )

    cur.executemany(insert_sql, data)
    conn.commit()

    cur.close()
    conn.close()

    logger.info("Table dropped, recreated, and data inserted successfully!")


def main():
    logger.info("Loading dataset")
    dataset = load_dataset("UCSC-VLAA/Recap-COCO-30K")

    save_dir = "./saved_images_coco_30k/"
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving images to {save_dir}")

    # Only process the first 100 images for testing and debugging
    # dataset["train"] = dataset["train"].select(range(100))
    dataset = dataset.map(lambda x: save_image_to_disk(x, save_dir))

    logger.info("Extracting images from dataset")
    images = [dataset["train"][i]["image"] for i in range(len(dataset["train"]))]
    logger.info(f"Total images: {len(images)}")

    device, tokenizer, processor, model = initialize_model()

    # image_arr = compute_embeddings(images, processor, model, device)
    image_arr = np.load("image_embeddings.npy")

    # Normalize embeddings
    image_arr = image_arr.T / np.linalg.norm(image_arr, axis=1)

    # Save embeddings to a .npy file
    embeddings_file = "image_embeddings.npy"
    np.save(embeddings_file, image_arr)
    logger.info(f"Embeddings saved to {embeddings_file}")

    # insert embeddings to database
    # logger.info("Inserting embeddings to database")
    # insert_to_db(dataset, image_arr)
    db = PostgreSQLDatabase("retrieval_db")
    db.insert_data(dataset, image_arr)


if __name__ == "__main__":
    logger.info("Starting main process")
    main()
    logger.info("Process completed")
