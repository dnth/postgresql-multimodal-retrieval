from pgmmr.database import PostgreSQLDatabase
from pgmmr.datasets import HuggingFaceDatasets
from pgmmr.models import CLIP


def main():
    # Get dataset
    ds = HuggingFaceDatasets("UCSC-VLAA/Recap-COCO-30K")
    ds.save_dataset_images("saved_images_coco_30k")
    image_filepaths = ds.get_image_paths()
    dataset_df = ds.to_pandas()

    # Load model
    clip = CLIP(model_id="openai/clip-vit-base-patch32")
    image_embeddings = clip.encode_image(image_filepaths, batch_size=256)

    # Load into database
    with PostgreSQLDatabase("retrieval_db") as db:
        db.insert_data(dataset_df, image_embeddings)

if __name__ == "__main__":
    main()
