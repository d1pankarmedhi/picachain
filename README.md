<div align="center">
<img src="https://github.com/d1pankarmedhi/picachain/assets/136924835/3a299c21-6590-4ee1-a3c1-73a92653f21e" height=150></img>
<h3>⚡️ Build quick ML pipelines for images</h3>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

[![PyPi license](https://badgen.net/pypi/license/pip/)]() [![PyPI version fury.io](https://badge.fury.io/py/picachain.svg)](https://pypi.python.org/pypi/picachain/)



</div>

## 📌 Install Picachain

```bash
pip install picachain
```

## 🔍️ Build a quick image search engine
Use **ChromaDB** or **Pinecone** for storage with **CLIP** embeddings.
Check out a demo [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FbruIGMBrD7VW5jCHStHzGlsEuigbS0q?usp=sharing)

```python
from PIL import Image
import matplotlib.pyplot as plt

# import from picachain
from picachain.datastore import ChromaStore
from picachain.embedding import ClipEmbedding
from picachain.retriever import ImageRetriever
from picachain.search import ImageSearch
```

```python
img = Image.open("image.png") # query image
images = [...] # list of images
```

```python
# initiate embedding, datastore and retriever
embedding = ClipEmbedding()
datastore = ChromaStore("test-collection")
retriever = ImageRetriever(datastore, embedding, images)
image_search = ImageSearch(retriever, embedding, img)
result = image_search.search_relevant_images(top_k=3) # get top 3 relevant images

for img, score in result: # [(img, score), (img, score)]
    plt.imshow(img)
    plt.show()

```
