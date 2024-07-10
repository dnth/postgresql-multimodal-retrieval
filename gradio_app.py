import argparse
import math
import os

import gradio as gr
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import psycopg
import torch
from loguru import logger
from pgvector.psycopg import register_vector
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast

def connect_to_database(dbname="retrieval_db"):
    conn = psycopg.connect(dbname=dbname, autocommit=True)
    register_vector(conn)
    return conn


def get_sql_query(num_results: int = 12):
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


def initialize_model(model_id="openai/clip-vit-base-patch32"):
    logger.info(f"Initializing model: {model_id}")
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
    axs = axs.flatten()

    output_images = []
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
        
        output_images.append((image_filepath, f"{image_filename} | {rrf_score:.4f}"))

    for j in range(i + 1, len(axs)):
        axs[j].axis("off")
        axs[j].set_visible(False)

    fig.suptitle("Retrieval Results (filename|RRF score)")
    plt.tight_layout(pad=4.0)
    plt.close(fig)

    return output_images

def image_retrieval(query, num_results=12):
    conn = connect_to_database()
    device, tokenizer, processor, model = initialize_model()
    
    text_emb = tokenize_text(query, tokenizer, model, device)

    k = 60
    sql = get_sql_query(num_results)
    results = execute_query(conn, sql, query, text_emb, k)

    return plot_results(results)



def gradio_interface(query, num_results):
    results = image_retrieval(query, num_results)
    images = [img[0] for img in results]
    captions = [img[1] for img in results]
    return images

# Set up the Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("Hybrid Search using CLIP and Keyword Search with RRF")
    gr.Markdown("Enter a text query to retrieve relevant images.")
    
    with gr.Row():
        query_input = gr.Textbox(label="Enter your query")
        num_results = gr.Slider(minimum=1, maximum=20, step=1, value=12, label="Number of results")
    submit_btn = gr.Button("Retrieve Images")
    
    gallery = gr.Gallery(
        label="Retrieved Images",
        show_label=True,
        columns=4,
        # rows=3,
        # height="auto",
        object_fit="contain"
    )
    
    submit_btn.click(
        fn=gradio_interface,
        inputs=[query_input, num_results],
        outputs=gallery
    )

if __name__ == "__main__":
    iface.launch(share=True)