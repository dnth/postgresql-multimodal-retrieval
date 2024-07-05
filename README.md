# postgresql-multimodal-retrieval
Multimodal retrieval using Vision Language Model with PostgreSQL database.


## Setting up

### Create a conda environment

```bash
conda create -n postgresql-multimodal python=3.10
conda activate postgresql-multimodal
```
### Install PostgreSQL

```bash
conda install -c conda-forge postgresql
psql --version
```

### Install pgvector

```bash
conda install -c conda-forge pgvector
```

### Initialize and start PostgreSQL

```bash
initdb -D mylocal_db
pg_ctl -D mylocal_db -l logfile start
```

### Create a database

```bash
createuser pokemon_user
createdb pokemon_db
```

## Install packages

```bash
pip install -r requirements.txt
```

## References

+ https://minimaxir.com/2024/06/pokemon-embeddings/
+ https://github.com/minimaxir/pokemon-embeddings/tree/main
+ https://towardsdatascience.com/quick-fire-guide-to-multi-modal-ml-with-openais-clip-2dad7e398ac0
+ https://github.com/pgvector/pgvector
+ https://huggingface.co/nomic-ai
+ https://huggingface.co/datasets/wanghaofan/pokemon-wiki-captions
  
