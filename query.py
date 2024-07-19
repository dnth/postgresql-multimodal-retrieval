import argparse

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import psycopg
from loguru import logger
from pgvector.psycopg import register_vector

import math

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from pgmmr.models import CLIP
from pgmmr.database import PostgreSQLDatabase

def connect_to_database(dbname="retrieval_db"):
    conn = psycopg.connect(dbname=dbname, autocommit=True)
    register_vector(conn)
    return conn


def hybrid_search(num_results: int = 12):
    return f"""
    WITH semantic_search AS (
        SELECT image_id, image_filepath, RANK () OVER (ORDER BY img_emb <=> %(embedding)s) AS rank
        FROM image_metadata
        ORDER BY img_emb <=> %(embedding)s
        LIMIT 20
    ),
    keyword_search AS (
        SELECT image_id, image_filepath, RANK () OVER (ORDER BY ts_rank_cd(to_tsvector('english', recaption), query) DESC)
        FROM image_metadata, plainto_tsquery('english', %(query)s) query
        WHERE to_tsvector('english', recaption) @@ query
        ORDER BY ts_rank_cd(to_tsvector('english', recaption), query) DESC
        LIMIT 20
    )
    SELECT
        COALESCE(semantic_search.image_id, keyword_search.image_id) AS id,
        COALESCE(semantic_search.image_filepath, keyword_search.image_filepath) AS image_filepath,
        COALESCE(1.0 / (%(k)s + semantic_search.rank), 0.0) +
        COALESCE(1.0 / (%(k)s + keyword_search.rank), 0.0) AS score
    FROM semantic_search
    FULL OUTER JOIN keyword_search ON semantic_search.image_id = keyword_search.image_id
    ORDER BY score DESC
    LIMIT {num_results}
    """


def tokenize_text(query, tokenizer, model, device):
    logger.info(f"Tokenizing text: {query}")
    inputs = tokenizer(query, return_tensors="pt").to(device)
    text_emb = model.get_text_features(**inputs)
    text_emb = text_emb.cpu().detach().numpy()
    return text_emb.flatten()


def execute_query(conn, sql, query, embedding, k):
    logger.info("Executing vector search")
    results = conn.execute(
        sql, {"query": query, "embedding": embedding, "k": k}
    ).fetchall()
    return results





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

    db = PostgreSQLDatabase("retrieval_db")
    db.connect()
    db.setup_pgvector_extension()

    clip = CLIP(model_id="openai/clip-vit-base-patch32", device="cpu")
    text_embeddings = clip.encode_text(args.query)

    sql = hybrid_search(args.num_results)
    results = execute_query(db.conn, sql, args.query, text_embeddings, k=60)

    plot_results(results)


if __name__ == "__main__":
    main()
