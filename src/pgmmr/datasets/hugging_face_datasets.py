import os

from loguru import logger

from datasets import load_dataset


class HuggingFaceDatasets:
    """
    Class to handle Hugging Face datasets, loading and processing them.
    """

    def __init__(
        self, dataset_name: str, num_images: int = None
    ):
        self.dataset_name = dataset_name
        logger.info(f"Loading dataset: {dataset_name}")

        self.dataset = load_dataset(
            self.dataset_name, split="all"
        )

        if num_images:
            logger.info(f"Subsetting dataset to {num_images} images")
            self.dataset = self.dataset.select(range(num_images))

    def save_dataset_images(self, save_dir: str):
        logger.info(f"Saving images to folder: {save_dir}")

        def save_image_to_disk(example, save_dir):
            filename = f"{example['image_id']}.jpg"
            filepath = os.path.join(save_dir, filename)
            example["image"].save(filepath)
            return {"image_filepath": filepath}

        os.makedirs(save_dir, exist_ok=True)
        self.dataset = self.dataset.map(
            lambda x: save_image_to_disk(x, save_dir), desc="Saving images", num_proc=14
        )

    def get_image_paths(self) -> list[str]:
        return self.dataset["image_filepath"]

    def to_pandas(self):
        self.dataset = self.dataset.remove_columns(["image"])
        return self.dataset.to_pandas()
