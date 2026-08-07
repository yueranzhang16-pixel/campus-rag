import unittest

import numpy as np

from campus_rag.embeddings import EmbeddingIndex
from campus_rag.retrieval import Chunk, ParentChunk


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

    def test_search_expands_parent_context(self):
        parent = ParentChunk("tree.md::树", "tree.md", "树", ["差异：", "数据位于叶子结点。"])
        index = EmbeddingIndex(
            chunks=[Chunk("tree.md", "差异：", "树 B+树", parent.id, 0)],
            vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
            parents={parent.id: parent},
            _encoder=FakeEncoder(),
        )

        result = index.search("树的差异", top_k=1)

        self.assertEqual(result[0].parent_text, "数据位于叶子结点。")

    def test_corpus_fingerprint_survives_embedding_index_serialization(self):
        index = EmbeddingIndex(
            chunks=[Chunk("tree.md", "tree", "tree")],
            vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
            corpus_fingerprint="abc123",
        )
        restored = EmbeddingIndex.from_dict(index.to_dict())
        self.assertEqual(restored.corpus_fingerprint, "abc123")
