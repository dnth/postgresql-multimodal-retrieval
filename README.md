# postgresql-multimodal-retrieval
Multimodal retrieval using Vision Language Model with PostgreSQL database - A full stack implementation.

+ Database: PostgreSQL
+ Vision Language Model: OpenAI CLIP (`transformers` implementation) 
+ Dataset: Hugging Face Datasets
+ Frontend: Flet / Gradio
+ Deployment: Docker
+ Infrastructure: Hugging Face Spaces

Features:
+ Image to image search
+ Text to image search
+ Hybrid search

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
createuser retrieval_user
createdb retrieval_db -O retrieval_user
```

### Install packages

```bash
pip install -r requirements.txt
```

## Usage

### Load Dataset into database

See [notebook](notebooks/load_dataset_into_postgres.ipynb).

Check database for inserted contents:

```
psql -d retrieval_db -U retrieval_user
```

```
select * image_metadata from  limit 5;
```

outputs

```
| name_en | name_zh | text_en | text_zh | image_filepath |
|---------|---------|---------|---------|----------------|
| abomasnow | 暴雪王 | Grass attributes,Blizzard King standing on two feet, with white fluff all over, lavender eyes, and a few long strips of fur covering its mouth | 草属性，双脚站立的暴雪王，全身白色的绒毛，淡紫色的眼睛，几缕长条装的毛皮盖着它的嘴巴 | ./saved_images/abomasnow_2058735963577945329.png |
| abra | 凯西 | Super power attributes, the whole body is yellow, the head shape is like a fox, with a pointed nose, three fingers on the hands and feet, and a brown ring at the end of the long tail | 超能力属性，通体黄色，头部外形类似狐狸，尖尖鼻子，手和脚上都有三个指头，长尾巴末端带着一个褐色圆环 | ./saved_images/abra_5265223410023725368.png |
| absol | 阿勃梭鲁 | Evil attribute, with white hair, blue-gray part without hair, bow-like horn on the right side of the head, red eyes | 恶属性，有白色毛发，没毛发的部分是蓝灰色，头右边类似弓的角，红色眼睛 | ./saved_images/absol_-8493509098281981613.png |
| accelgor | 敏捷虫 | Insect attributes,upright agile insects, the whole body is covered by mucous membranes, the neck is gray, the head has a four-pointed star shape, and there is a belt with eyes on the back | 虫属性，直立型的敏捷虫，全身被粘膜包裹着，脖子下是灰色，头部有一个四角星形状，背后有眼神出去的带子 | ./saved_images/accelgor_-4254867256317794903.png |
| aegislash-shield | 坚盾剑怪 | Steel attribute, huge sword body, hilt, sword tan, and golden-yellow sword spine, deep purple eyes and palm, black arms, jagged blade | 钢属性，巨大的剑身，剑柄，剑镡，和金黄色剑脊，深紫色眼睛和手掌，黑色手臂，锯齿状剑锋 | ./saved_images/aegislash-shield_-9192879603048663302.png |

```

### Compute embeddings
Run 

```
python compute_embeddings.py
```

This will compute embeddings for all the images in the database and store them in the `image_embeddings.npy` numpy array.

### Query
See [notebook](notebooks/query.ipynb).

## References

+ https://minimaxir.com/2024/06/pokemon-embeddings/
+ https://github.com/minimaxir/pokemon-embeddings/tree/main
+ https://towardsdatascience.com/quick-fire-guide-to-multi-modal-ml-with-openais-clip-2dad7e398ac0
+ https://github.com/pgvector/pgvector
+ https://huggingface.co/nomic-ai
+ https://huggingface.co/datasets/wanghaofan/pokemon-wiki-captions
https://github.com/jamescalam/clip-demos/blob/main/tutorials/00-get-started/00-get-started.ipynb
  
