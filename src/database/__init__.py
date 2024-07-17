import psycopg
from loguru import logger
import datasets
import numpy as np
from pgvector.psycopg import register_vector

class PostgreSQLDatabase:
    def __init__(self, database_name: str) -> None:
        self.database_name = database_name
        self.conn = None
        self.cur = None

    def connect(self):
        try:
            self.conn = psycopg.connect(dbname=self.database_name)
            self.cur = self.conn.cursor()
            logger.info("Connected to database")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
        
    
    def disconnect(self):
        try:
            self.cur.close()
            self.conn.close()
            logger.info("Disconnected from database")
        except Exception as e:
            logger.error(f"Error disconnecting from database: {e}")

    def setup_pgvector_extension(self):
        try:
            self.cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(self.conn)
            logger.info("pgvector extension initialized")
        except Exception as e:
            logger.error(f"Error creating pgvector extension: {e}")

    def create_table(self):
        try:
            self.cur.execute("""
            DROP TABLE IF EXISTS image_metadata;

            CREATE TABLE image_metadata (
                image_id INTEGER PRIMARY KEY,
                coco_url TEXT,
                caption TEXT,
                recaption TEXT,
                image_filepath TEXT,
                img_emb vector(512)
            )
            """)
            logger.info("Table created")
        except Exception as e:
            logger.error(f"Error creating table: {e}")
            raise

    def insert_data(self, dataset: datasets.Dataset, embeddings: np.ndarray):
        self.connect()
        self.setup_pgvector_extension()
        self.create_table()

        df = dataset["train"].to_pandas()
        df = df.drop(columns=["image"])

        df["image_filepath"] = df["image_filepath"].apply(lambda x: x.split("/")[-1])

        df["img_emb"] = embeddings.T.tolist()

        # Prepare the insert statement
        insert_sql = """
        INSERT INTO image_metadata (image_id, coco_url, caption, recaption, image_filepath, img_emb)
        VALUES (%s, %s, %s, %s, %s, %s)
        """

        # Convert dataframe to list of tuples
        data = []
        for _, row in df.iterrows():
            data.append(
                (
                    row["image_id"],
                    row["coco_url"],
                    row["caption"],
                    row["recaption"],
                    row["image_filepath"],
                    row["img_emb"],
                )
            )

        self.cur.executemany(insert_sql, data)
        self.conn.commit()

        self.disconnect()

        logger.info("Data inserted successfully!")
        

