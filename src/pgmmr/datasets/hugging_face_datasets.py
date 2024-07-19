import os

from datasets import load_dataset


from loguru import logger


class HuggingFaceDatasets:
    """
    Class to handle Hugging Face datasets, loading and processing them.
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        logger.info(f"Loading dataset: {dataset_name}")

        self.dataset = load_dataset(self.dataset_name, split="all")

    def save_images(self, save_dir: str, num_images: int = None):
        logger.info(f"Saving images to folder: {save_dir}")
        if num_images is not None:
            logger.info(f"Saving a subset of {num_images} images")

        def save_image_to_disk(example, save_dir):
            filename = f"{example['image_id']}.jpg"
            filepath = os.path.join(save_dir, filename)
            example["image"].save(filepath)
            return {"image_filepath": filepath}

        os.makedirs(save_dir, exist_ok=True)

        if num_images is None or num_images >= len(self.dataset):
            num_images = len(self.dataset)

        self.dataset = self.dataset.select(list(range(num_images))).map(
            lambda x: save_image_to_disk(x, save_dir), desc="Saving images", num_proc=14
        )

    @property
    def image_paths(self) -> list[str]:
        return self.dataset["image_filepath"]

    @property
    def pandas_df(self):
        logger.info("Converting dataset to pandas DataFrame")

        columns = [col for col in self.dataset.column_names if col != "image"]

        try:
            return self.dataset.select_columns(columns).to_pandas()
        except MemoryError:
            logger.error(
                "MemoryError: The dataset is too large to convert to a DataFrame at once. "
                "Consider using the to_pandas method with chunking instead."
            )
            raise
