import unittest

import numpy as np

from campus_rag.embeddings import EmbeddingIndex
from campus_rag.retrieval import Chunk


class FakeEncoder:
    def encode(self, texts, **_kwargs):
        vectors = []
        for text in texts:
            vectors.append([1.0, 0.0] if "树" in text else [0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


class EmbeddingIndexTests(unittest.TestCase):
    def test_search_ranks_the_nearest_vector_first(self):
        index = EmbeddingIndex(
            chunks=[Chunk("tree.md", "树的遍历", "树"), Chunk("array.md", "数组存储", "数组")],
            vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            _encoder=FakeEncoder(),
        )
        result = index.search("树有哪些遍历方式", top_k=1)
        self.assertEqual(result[0].source, "tree.md")
