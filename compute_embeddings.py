from pgmmr.database import PostgreSQLDatabase
from pgmmr.datasets import HuggingFaceDatasets
from pgmmr.models import CLIP


def main():
    # Get dataset
    ds = HuggingFaceDatasets("UCSC-VLAA/Recap-COCO-30K")
    ds.save_images("saved_images_coco_30k", num_images=1000)

    # Load model
    clip = CLIP(model_id="openai/clip-vit-base-patch32")
    image_embeddings = clip.encode_image(ds.image_paths, batch_size=256)

    # Load into database
    with PostgreSQLDatabase("retrieval_db") as db:
        db.insert_data(ds.pandas_df, image_embeddings)


if __name__ == "__main__":
    main()
