<div align="center">
<img src="https://github.com/d1pankarmedhi/picachain/assets/136924835/3a299c21-6590-4ee1-a3c1-73a92653f21e" height=150></img>
<h3>⚡️ A Simple ready-to-use Image search engine library</h3>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

[![PyPi license](https://badgen.net/pypi/license/pip/)]() [![PyPI version fury.io](https://badge.fury.io/py/picachain.svg)](https://pypi.python.org/pypi/picachain/)



</div>

## 📌 Install Picachain

```bash
pip install picachain
```

## 🥇 Demo
Check out **Picachain** and **ChromaDB** demo [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FbruIGMBrD7VW5jCHStHzGlsEuigbS0q?usp=sharing)

## 🚀 Getting Started
Create your own image search pipeline with just a few lines of code.

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
img = Image.open("image-path") # query image
images = [Image.open(os.path.join("images-path", image)) for image in os.listdir("images-path")] # image collection
```

```python
# initiate embedding, datastore and retriever
embedding = ClipEmbedding()
datastore = ChromaStore("test-collection")
retriever = ImageRetriever(datastore, embedding, images)

image_search = ImageSearch(retriever=retriever, embedding=embedding, query_img=img)
result = image_search.search_relevant_images(top_k=3) # get top 3 relevant images

for img, score in result: # [(img, score), (img, score)]
    plt.imshow(img)
    plt.show()

```

It is under continuous development so currently supports only [ChromaDB](https://docs.trychroma.com/). We are working on integrating all popular vector databases such as [Pinecone](https://www.pinecone.io/), [Weaviate](https://weaviate.io/), etc. 






