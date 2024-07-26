import argparse
import math

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


from pgmmr.database import PostgreSQLDatabase
from pgmmr.models import CLIP

from pgmmr.retrievers.hybrid_search import HybridSearch


def plot_results(results, image_dir="./saved_images_coco_30k/"):
    num_images = len(results)
    num_cols = 4
    num_rows = math.ceil(num_images / num_cols)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axs = axs.flatten()  # Flatten the 2D array of axes to make indexing easier

    for i, row in enumerate(results):
        if i >= len(axs):
            break

        image_filename = row[1]
        rrf_score = row[2]
        image_filepath = f"{image_dir}{image_filename}"
        img = mpimg.imread(image_filepath)
        axs[i].imshow(img)
        axs[i].axis("off")
        axs[i].set_title(f"{image_filename} | {rrf_score:.4f}", fontsize=10)

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")
        axs[j].set_visible(False)

    fig.suptitle("Retrieval Results (filename | RRF score)")
    plt.tight_layout(pad=4.0)
    plt.savefig("images/results.png")

    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Image retrieval based on text query")
    parser.add_argument("query", type=str, help="Text query for image retrieval")
    parser.add_argument(
        "--num_results", type=int, default=12, help="Number of images to retrieve"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    clip = CLIP(model_id="openai/clip-vit-base-patch32", device="cpu")
    input_text_embeddings = clip.encode_text(args.query)

    with PostgreSQLDatabase("retrieval_db") as db:
        search_engine = HybridSearch(db.conn, num_results=args.num_results)
        results = search_engine.search(args.query, input_text_embeddings)

    plot_results(results)


if __name__ == "__main__":
    main()
