from loguru import logger
from typing import List
import time
from .result import Result


class HybridSearch:
    """
    Hybrid search class that combines caption vector search and caption keyword search
    """

    def __init__(self, conn, model, num_results: int = 12, k: int = 60):
        self.conn = conn
        self.model = model

        self.num_results = num_results
        self.k = k

    def build_search_query(self) -> str:
        return f"""
        WITH text_to_image_vector_search AS (
            SELECT image_id, image_filepath, RANK () OVER (ORDER BY img_emb <=> %(embedding)s) AS rank
            FROM image_metadata
            ORDER BY img_emb <=> %(embedding)s
            LIMIT {self.num_results}
        ),
        caption_keyword_search AS (
            SELECT image_id, image_filepath, RANK () OVER (ORDER BY ts_rank_cd(to_tsvector('english', recaption), query) DESC)
            FROM image_metadata, plainto_tsquery('english', %(query)s) query
            WHERE to_tsvector('english', recaption) @@ query
            ORDER BY ts_rank_cd(to_tsvector('english', recaption), query) DESC
            LIMIT {self.num_results}
        )
        SELECT
            COALESCE(text_to_image_vector_search.image_id, caption_keyword_search.image_id) AS id,
            COALESCE(text_to_image_vector_search.image_filepath, caption_keyword_search.image_filepath) AS image_filepath,
            COALESCE(1.0 / (%(k)s + text_to_image_vector_search.rank), 0.0) +
            COALESCE(1.0 / (%(k)s + caption_keyword_search.rank), 0.0) AS score
        FROM text_to_image_vector_search
        FULL OUTER JOIN caption_keyword_search ON text_to_image_vector_search.image_id = caption_keyword_search.image_id
        ORDER BY score DESC
        LIMIT {self.num_results}
        """

    def search(self, query: str) -> List[Result]:
        logger.info("Executing search")
        try:
            input_text_embeddings = self.model.encode_text(query)
            sql = self.build_search_query()
            t = time.time()
            results = self.conn.execute(
                sql, {"query": query, "embedding": input_text_embeddings, "k": self.k}
            ).fetchall()

            logger.info(f"Search executed in {time.time() - t:.4f} seconds")
            results = [
                Result(id=row[0], image_filename=row[1], rrf_score=row[2])
                for row in results
            ]

            return results
        except Exception as e:
            logger.error(f"Error executing search: {str(e)}")
            raise
