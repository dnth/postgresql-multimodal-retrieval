from loguru import logger
from typing import List, Tuple, Any


class HybridSearch:
    def __init__(self, conn, model, num_results: int = 12, k: int = 60):
        self.conn = conn
        self.model = model

        self.num_results = num_results
        self.k = k

    def build_search_query(self) -> str:
        return f"""
        WITH semantic_search AS (
            SELECT image_id, image_filepath, RANK () OVER (ORDER BY img_emb <=> %(embedding)s) AS rank
            FROM image_metadata
            ORDER BY img_emb <=> %(embedding)s
            LIMIT {self.num_results}
        ),
        keyword_search AS (
            SELECT image_id, image_filepath, RANK () OVER (ORDER BY ts_rank_cd(to_tsvector('english', recaption), query) DESC)
            FROM image_metadata, plainto_tsquery('english', %(query)s) query
            WHERE to_tsvector('english', recaption) @@ query
            ORDER BY ts_rank_cd(to_tsvector('english', recaption), query) DESC
            LIMIT {self.num_results}
        )
        SELECT
            COALESCE(semantic_search.image_id, keyword_search.image_id) AS id,
            COALESCE(semantic_search.image_filepath, keyword_search.image_filepath) AS image_filepath,
            COALESCE(1.0 / (%(k)s + semantic_search.rank), 0.0) +
            COALESCE(1.0 / (%(k)s + keyword_search.rank), 0.0) AS score
        FROM semantic_search
        FULL OUTER JOIN keyword_search ON semantic_search.image_id = keyword_search.image_id
        ORDER BY score DESC
        LIMIT {self.num_results}
        """

    def search(self, query: str) -> List[Tuple[Any]]:
        logger.info("Executing search")
        try:
            input_text_embeddings = self.model.encode_text(query)
            sql = self.build_search_query()
            results = self.conn.execute(
                sql, {"query": query, "embedding": input_text_embeddings, "k": self.k}
            ).fetchall()

            return results
        except Exception as e:
            logger.error(f"Error executing search: {str(e)}")
            raise
