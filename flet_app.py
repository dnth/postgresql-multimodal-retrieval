import argparse
import math
import os

import flet as ft
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

def main(page: ft.Page):
    page.title = "Hybrid Search using CLIP and Keyword Search with RRF"
    page.theme_mode = "light"

    def search_images(e):
        if not query_input.value:
            query_input.error_text = "Query cannot be empty"
            page.update()
            return

        query_input.error_text = None  # Clear any previous error
        query = query_input.value
        num_results = int(num_results_slider.value)
        results = image_retrieval(query, num_results)
        
        image_gallery.controls.clear()
        for img_path, caption in results:
            image_gallery.controls.append(
                ft.Image(
                    src=img_path,
                    width=200,
                    height=200,
                    fit=ft.ImageFit.CONTAIN,
                    tooltip=caption
                )
            )
        page.update()

    def update_slider_value(e):
        slider_value.value = f"Number of results: {int(e.control.value)}"
        page.update()

    query_input = ft.TextField(
        label="Enter your query",
        hint_text="Type your search query here",
        on_change=lambda _: setattr(query_input, 'error_text', None)
    )
    num_results_slider = ft.Slider(
        min=1, 
        max=20, 
        value=12, 
        label="Number of results",
        on_change=update_slider_value
    )
    slider_value = ft.Text(f"Number of results: {int(num_results_slider.value)}")
    submit_btn = ft.ElevatedButton("Retrieve Images", on_click=search_images)
    
    image_gallery = ft.Row(
        wrap=True,
        scroll="auto",
        expand=True,
        alignment=ft.MainAxisAlignment.CENTER
    )

    page.add(
        ft.Text("Hybrid Search using CLIP and Keyword Search with RRF", size=20),
        query_input,
        ft.Row([num_results_slider, slider_value]),
        submit_btn,
        image_gallery
    )

if __name__ == "__main__":
    ft.app(target=main)