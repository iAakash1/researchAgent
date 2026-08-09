"""Semantic, BM25 and hybrid retrieval.

No test requires a running Qdrant or Ollama: the in-memory vector store implements the
full port with exact cosine search, and embeddings are faked deterministically. The
Qdrant adapter's own behaviour is covered by contract tests that skip when it is absent.
"""

from __future__ import annotations

import math
from typing import ClassVar

import pytest

from researchagent.config.schemas import FusionSettings, FusionStrategy
from researchagent.core.exceptions import (
    EmbeddingError,
    IndexIncompatibleError,
)
from researchagent.core.interfaces.embeddings import (
    EmbeddingHealth,
    EmbeddingModel,
    ModelIdentity,
)
from researchagent.core.interfaces.retrieval import (
    KnowledgeRetriever,
    RetrievalHit,
    RetrievalLayer,
    RetrievalResult,
)
from researchagent.core.interfaces.vector_store import VectorFilter
from researchagent.core.validation import Confidence, ConfidenceSignal, ValidationResult
from researchagent.evaluation.metrics import (
    evaluate,
    mean_metrics,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from researchagent.integrations.memory_store import InMemoryVectorStore
from researchagent.models.knowledge import (
    DatasetDetails,
    KnowledgeKind,
    KnowledgeObject,
    MethodDetails,
    PaperKnowledge,
)
from researchagent.models.query import ResearchQuery
from researchagent.repositories.knowledge_repository import JsonKnowledgeRepository
from researchagent.schemas.knowledge import ValidatedKnowledge
from researchagent.services.retrieval import (
    BM25KnowledgeRetriever,
    ComponentRole,
    HybridKnowledgeRetriever,
    KnowledgeIndexer,
    RetrieverComponent,
    SemanticKnowledgeRetriever,
    represent,
)

QUOTE = "Metastable failures are triggered by sustained overload in distributed systems."


def an_object(
    name: str,
    *,
    kind: KnowledgeKind = KnowledgeKind.METHOD,
    paper: str = "manual:01",
    description: str = "",
    confidence: float = 0.8,
) -> KnowledgeObject:
    from researchagent.core.evidence import Evidence, SourceLocation

    return KnowledgeObject.model_validate(
        {
            "id": f"{paper}#{kind.value}:{name}",
            "kind": kind,
            "paper_id": paper,
            "name": name,
            "description": description,
            "details": MethodDetails() if kind is KnowledgeKind.METHOD else DatasetDetails(),
            "evidence": (
                Evidence.from_text(
                    claim="test",
                    quote=QUOTE,
                    location=SourceLocation(
                        document_id=paper,
                        page=4,
                        section_id="s004",
                        section_title="Results",
                        paragraph_index=2,
                    ),
                    produced_by="test",
                ),
            ),
            "confidence": Confidence.from_signals(
                [ConfidenceSignal(name="t", value=confidence, observation="fixture")]
            ),
            "extracted_by": "test",
        }
    )


class FakeEmbeddingModel(EmbeddingModel):
    """Deterministic bag-of-words vectors: no Ollama, but real cosine geometry.

    Similar texts genuinely produce similar vectors, so semantic retrieval is exercised
    rather than mocked away.
    """

    name: ClassVar[str] = "fake"

    def __init__(
        self, *, dimension: int = 64, fail: bool = False, model: str = "fake-embed"
    ) -> None:
        self._dimension = dimension
        self._fail = fail
        self._model = model
        self.calls = 0

    async def embed_text(self, text: str) -> tuple[float, ...]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[tuple[float, ...]]:
        if self._fail:
            raise EmbeddingError("embedding backend down", model=self._model)
        self.calls += 1
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> tuple[float, ...]:
        buckets = [0.0] * self._dimension
        for token in text.lower().split():
            buckets[hash(token) % self._dimension] += 1.0
        norm = math.sqrt(sum(value * value for value in buckets)) or 1.0
        return tuple(value / norm for value in buckets)

    def dimension(self) -> int:
        return self._dimension

    def model_identity(self) -> ModelIdentity:
        return ModelIdentity(provider=self.name, model_name=self._model, dimension=self._dimension)

    async def health(self) -> EmbeddingHealth:
        return EmbeddingHealth(healthy=not self._fail, identity=self.model_identity())

    async def aclose(self) -> None:
        return None


class StubRetriever(KnowledgeRetriever):
    """Returns a fixed ranking, or reports itself unavailable."""

    name: ClassVar[str] = "stub"

    def __init__(
        self, objects: list[KnowledgeObject], *, label: str = "stub", degraded: bool = False
    ) -> None:
        self._objects = objects
        self._label = label
        self._degraded = degraded

    async def retrieve(self, query: ResearchQuery) -> RetrievalResult[KnowledgeObject]:
        if self._degraded:
            return RetrievalResult[KnowledgeObject].unavailable(
                layer=RetrievalLayer.KNOWLEDGE,
                query=query,
                retrieved_by=self._label,
                reason="backend down",
            )
        hits = tuple(
            RetrievalHit[KnowledgeObject](
                item=obj,
                score=round(1.0 - index * 0.1, 6),
                signals=(
                    ConfidenceSignal(
                        name="stub", value=1.0 - index * 0.1, observation=f"rank {index + 1}"
                    ),
                ),
                retrieved_by=self._label,
            )
            for index, obj in enumerate(self._objects)
        )
        return RetrievalResult[KnowledgeObject](
            layer=RetrievalLayer.KNOWLEDGE,
            query=query,
            hits=hits,
            considered=len(self._objects),
            retrieved_by=self._label,
        )

    async def health(self) -> bool:
        return not self._degraded


async def seed(repository: JsonKnowledgeRepository, *objects: KnowledgeObject) -> None:
    by_paper: dict[str, list[KnowledgeObject]] = {}
    for item in objects:
        by_paper.setdefault(item.paper_id, []).append(item)
    for paper_id, items in by_paper.items():
        await repository.save(
            ValidatedKnowledge(
                value=PaperKnowledge(
                    paper_id=paper_id, document_sha256="abc", objects=tuple(items)
                ),
                validation=ValidationResult.passed(
                    validator="v",
                    subject_id=paper_id,
                    subject_type="PaperKnowledge",
                    confidence=Confidence.unknown(),
                ),
            )
        )


class TestRepresentation:
    def test_one_representation_serves_bm25_and_embeddings(self) -> None:
        """Two subsystems indexing different text would make their scores incomparable."""
        obj = an_object("Circuit Breaker Pattern", description="Trips on repeated failure.")

        text = represent(obj).text

        assert "Circuit Breaker Pattern" in text
        assert "method" in text
        assert "Trips on repeated failure." in text
        assert QUOTE in text


class TestInMemoryVectorStore:
    async def test_exact_cosine_ranking(self) -> None:
        store = InMemoryVectorStore()
        identity = ModelIdentity(provider="t", model_name="m", dimension=2)
        await store.ensure_collection(identity, "v1")
        await store.upsert(
            [
                _record("a", (1.0, 0.0), identity),
                _record("b", (0.0, 1.0), identity),
                _record("c", (0.7071, 0.7071), identity),
            ]
        )

        hits = await store.search((1.0, 0.0), limit=3)

        assert [hit.id for hit in hits] == ["a", "c", "b"]
        assert hits[0].score == pytest.approx(1.0, abs=1e-6)

    async def test_filters_apply_before_similarity(self) -> None:
        store = InMemoryVectorStore()
        identity = ModelIdentity(provider="t", model_name="m", dimension=2)
        await store.ensure_collection(identity, "v1")
        await store.upsert(
            [
                _record("a", (1.0, 0.0), identity, paper="manual:01"),
                _record("b", (1.0, 0.0), identity, paper="manual:02"),
            ]
        )

        hits = await store.search(
            (1.0, 0.0), limit=5, filters=VectorFilter(paper_ids=("manual:02",))
        )

        assert [hit.id for hit in hits] == ["b"]

    async def test_mixing_vector_spaces_is_refused(self) -> None:
        """Vectors from two models are not comparable; serving them together is worse
        than serving nothing."""
        store = InMemoryVectorStore()
        first = ModelIdentity(provider="t", model_name="a", dimension=2)
        second = ModelIdentity(provider="t", model_name="b", dimension=2)
        await store.ensure_collection(first, "v1")

        with pytest.raises(IndexIncompatibleError):
            await store.ensure_collection(second, "v1")

    async def test_upsert_rejects_foreign_vectors(self) -> None:
        store = InMemoryVectorStore()
        identity = ModelIdentity(provider="t", model_name="a", dimension=2)
        other = ModelIdentity(provider="t", model_name="b", dimension=2)
        await store.ensure_collection(identity, "v1")

        with pytest.raises(IndexIncompatibleError):
            await store.upsert([_record("x", (1.0, 0.0), other)])


class TestIndexing:
    async def test_index_preserves_recovery_metadata(
        self, knowledge_repository: JsonKnowledgeRepository
    ) -> None:
        """Every vector must be able to name what it represents and what produced it."""
        await seed(knowledge_repository, an_object("Circuit Breaker Pattern"))
        store = InMemoryVectorStore()
        indexer = KnowledgeIndexer(FakeEmbeddingModel(), store, knowledge_repository)

        report = await indexer.build()
        hits = await store.search(await FakeEmbeddingModel().embed_text("circuit"), limit=5)

        assert report.succeeded and report.objects_indexed == 1
        metadata = hits[0].metadata
        assert metadata.knowledge_object_id.endswith("Circuit Breaker Pattern")
        assert metadata.paper_id == "manual:01"
        assert metadata.kind is KnowledgeKind.METHOD
        assert metadata.evidence_ids
        assert metadata.page == 4
        assert metadata.section_title == "Results"
        assert metadata.model_identity.model_name == "fake-embed"
        assert metadata.index_version == report.index_version

    async def test_index_version_changes_with_the_model(
        self, knowledge_repository: JsonKnowledgeRepository
    ) -> None:
        """Changing the embedding model must never silently reuse vectors."""
        store = InMemoryVectorStore()
        first = KnowledgeIndexer(FakeEmbeddingModel(model="model-a"), store, knowledge_repository)
        second = KnowledgeIndexer(FakeEmbeddingModel(model="model-b"), store, knowledge_repository)

        assert first.index_version(
            FakeEmbeddingModel(model="model-a").model_identity()
        ) != second.index_version(FakeEmbeddingModel(model="model-b").model_identity())

    async def test_index_version_changes_with_dimension(
        self, knowledge_repository: JsonKnowledgeRepository
    ) -> None:
        indexer = KnowledgeIndexer(
            FakeEmbeddingModel(), InMemoryVectorStore(), knowledge_repository
        )

        small = indexer.index_version(ModelIdentity(provider="t", model_name="m", dimension=64))
        large = indexer.index_version(ModelIdentity(provider="t", model_name="m", dimension=768))

        assert small != large

    async def test_untrusted_knowledge_is_never_indexed(
        self, knowledge_repository: JsonKnowledgeRepository
    ) -> None:
        await knowledge_repository.save(
            ValidatedKnowledge(
                value=PaperKnowledge(
                    paper_id="manual:01",
                    document_sha256="abc",
                    objects=(an_object("Rejected"),),
                ),
                validation=ValidationResult.failed(
                    validator="v", subject_id="manual:01", subject_type="PaperKnowledge", issues=[]
                ),
            )
        )
        indexer = KnowledgeIndexer(
            FakeEmbeddingModel(), InMemoryVectorStore(), knowledge_repository
        )

        report = await indexer.build()

        assert report.objects_indexed == 0
        assert "manual:01" in report.papers_skipped

    async def test_indexing_degrades_when_embeddings_are_down(
        self, knowledge_repository: JsonKnowledgeRepository
    ) -> None:
        await seed(knowledge_repository, an_object("Anything"))
        indexer = KnowledgeIndexer(
            FakeEmbeddingModel(fail=True), InMemoryVectorStore(), knowledge_repository
        )

        report = await indexer.build()

        assert report.succeeded is False
        assert report.error is not None and "embedding_error" in report.error

    async def test_the_original_object_is_never_mutated(
        self, knowledge_repository: JsonKnowledgeRepository
    ) -> None:
        obj = an_object("Circuit Breaker Pattern")
        await seed(knowledge_repository, obj)
        indexer = KnowledgeIndexer(
            FakeEmbeddingModel(), InMemoryVectorStore(), knowledge_repository
        )

        await indexer.build()

        stored = await knowledge_repository.get("manual:01")
        assert stored is not None
        assert stored.value.objects[0] == obj


class TestBM25:
    async def test_ranks_by_term_specificity(
        self, knowledge_repository: JsonKnowledgeRepository
    ) -> None:
        await seed(
            knowledge_repository,
            an_object("Circuit Breaker Pattern", description="circuit breaker trips"),
            an_object("Autoscaling", description="adds capacity"),
            an_object("Prioritization", description="orders work"),
        )
        retriever = BM25KnowledgeRetriever(knowledge_repository)

        result = await retriever.retrieve(ResearchQuery(text="circuit breaker", limit=3))

        assert result.hits[0].item.name == "Circuit Breaker Pattern"
        assert result.hits[0].score == pytest.approx(1.0)

    async def test_scores_are_explained(
        self, knowledge_repository: JsonKnowledgeRepository
    ) -> None:
        await seed(knowledge_repository, an_object("Circuit Breaker Pattern"))
        retriever = BM25KnowledgeRetriever(knowledge_repository)

        result = await retriever.retrieve(ResearchQuery(text="circuit breaker"))

        assert "BM25 score" in result.hits[0].explain()

    async def test_no_match_is_an_empty_not_degraded_result(
        self, knowledge_repository: JsonKnowledgeRepository
    ) -> None:
        await seed(knowledge_repository, an_object("Autoscaling"))
        retriever = BM25KnowledgeRetriever(knowledge_repository)

        result = await retriever.retrieve(ResearchQuery(text="zzz unmatchable qqq"))

        assert result.is_empty
        assert result.is_usable is True


class TestSemanticRetrieval:
    async def _ready(
        self, repository: JsonKnowledgeRepository, *objects: KnowledgeObject
    ) -> SemanticKnowledgeRetriever:
        await seed(repository, *objects)
        store = InMemoryVectorStore()
        embeddings = FakeEmbeddingModel()
        await KnowledgeIndexer(embeddings, store, repository).build()
        return SemanticKnowledgeRetriever(embeddings, store, repository)

    async def test_retrieves_semantically_close_objects(
        self, knowledge_repository: JsonKnowledgeRepository
    ) -> None:
        retriever = await self._ready(
            knowledge_repository,
            an_object("Circuit Breaker Pattern", description="trips on repeated failure"),
            an_object("Pottery Glazing", description="ceramics kiln technique"),
        )

        result = await retriever.retrieve(
            ResearchQuery(text="Circuit Breaker Pattern trips on repeated failure", limit=2)
        )

        assert result.hits[0].item.name == "Circuit Breaker Pattern"
        assert "cosine" in result.hits[0].explain()

    async def test_a_vector_hit_resolves_through_the_repository(
        self, knowledge_repository: JsonKnowledgeRepository
    ) -> None:
        """The vector store is not the source of truth."""
        retriever = await self._ready(knowledge_repository, an_object("Circuit Breaker Pattern"))

        result = await retriever.retrieve(ResearchQuery(text="circuit breaker"))

        assert isinstance(result.hits[0].item, KnowledgeObject)
        assert result.hits[0].item.evidence  # full object, not a payload copy

    async def test_reports_degraded_when_embeddings_are_down(
        self, knowledge_repository: JsonKnowledgeRepository
    ) -> None:
        """An outage must be distinguishable from an empty corpus."""
        await seed(knowledge_repository, an_object("Circuit Breaker Pattern"))
        retriever = SemanticKnowledgeRetriever(
            FakeEmbeddingModel(fail=True), InMemoryVectorStore(), knowledge_repository
        )

        result = await retriever.retrieve(ResearchQuery(text="circuit breaker"))

        assert result.is_empty
        assert result.degraded is True
        assert result.unavailable_reason is not None

    async def test_health_reflects_both_dependencies(
        self, knowledge_repository: JsonKnowledgeRepository
    ) -> None:
        healthy = SemanticKnowledgeRetriever(
            FakeEmbeddingModel(), InMemoryVectorStore(), knowledge_repository
        )
        broken = SemanticKnowledgeRetriever(
            FakeEmbeddingModel(fail=True), InMemoryVectorStore(), knowledge_repository
        )

        assert await healthy.health() is True
        assert await broken.health() is False


class TestFusion:
    def _objects(self) -> list[KnowledgeObject]:
        return [an_object(f"Object {index}") for index in range(4)]

    async def test_rrf_combines_rankings(self) -> None:
        objects = self._objects()
        hybrid = HybridKnowledgeRetriever(
            [
                RetrieverComponent(StubRetriever(objects, label="a"), ComponentRole.LEXICAL, 1.0),
                RetrieverComponent(
                    StubRetriever(list(reversed(objects)), label="b"), ComponentRole.DENSE, 1.0
                ),
            ],
            FusionSettings(strategy=FusionStrategy.RECIPROCAL_RANK),
        )

        result = await hybrid.retrieve(ResearchQuery(text="anything", limit=4))

        assert len(result.hits) == 4
        assert all(0.0 <= hit.score <= 1.0 for hit in result.hits)
        # Every hit explains which component contributed and at what rank.
        assert any("ranked #" in signal.observation for signal in result.hits[0].signals)

    async def test_weighted_fusion_respects_weights(self) -> None:
        objects = self._objects()
        hybrid = HybridKnowledgeRetriever(
            [
                RetrieverComponent(
                    StubRetriever(objects, label="strong"), ComponentRole.LEXICAL, 10.0
                ),
                RetrieverComponent(
                    StubRetriever(list(reversed(objects)), label="weak"), ComponentRole.DENSE, 0.1
                ),
            ],
            FusionSettings(strategy=FusionStrategy.WEIGHTED_SCORE),
        )

        result = await hybrid.retrieve(ResearchQuery(text="anything", limit=4))

        assert result.hits[0].item.name == objects[0].name

    async def test_a_degraded_component_is_excluded_not_counted_as_zero(self) -> None:
        """An outage must not push every semantically-strong result down the ranking."""
        objects = self._objects()
        hybrid = HybridKnowledgeRetriever(
            [
                RetrieverComponent(StubRetriever(objects, label="ok"), ComponentRole.LEXICAL, 1.0),
                RetrieverComponent(
                    StubRetriever([], label="down", degraded=True), ComponentRole.DENSE, 1.0
                ),
            ]
        )

        result = await hybrid.retrieve(ResearchQuery(text="anything", limit=4))

        assert result.hits
        assert result.degraded is True  # the caller is told it was diminished
        assert result.unavailable_reason is not None and "down" in result.unavailable_reason

    async def test_every_component_down_is_an_unavailable_result(self) -> None:
        hybrid = HybridKnowledgeRetriever(
            [
                RetrieverComponent(
                    StubRetriever([], label="a", degraded=True), ComponentRole.LEXICAL, 1.0
                ),
            ]
        )

        result = await hybrid.retrieve(ResearchQuery(text="anything"))

        assert result.is_usable is False
        assert result.is_empty

    async def test_hybrid_requires_a_component(self) -> None:
        with pytest.raises(ValueError, match="at least one component"):
            HybridKnowledgeRetriever([])

    async def test_hybrid_is_itself_a_knowledge_retriever(self) -> None:
        """Composable: v0.8 adds a graph retriever to the mix without changing callers."""
        hybrid = HybridKnowledgeRetriever(
            [RetrieverComponent(StubRetriever([]), ComponentRole.LEXICAL, 1.0)]
        )

        assert isinstance(hybrid, KnowledgeRetriever)
        assert hybrid.layer is RetrievalLayer.KNOWLEDGE


class TestMetrics:
    def test_precision_and_recall(self) -> None:
        retrieved = ["a", "x", "b", "y"]
        relevant = {"a", "b", "c"}

        assert precision_at_k(retrieved, relevant, 2) == 0.5
        assert recall_at_k(retrieved, relevant, 4) == pytest.approx(2 / 3)

    def test_mrr_rewards_an_early_hit(self) -> None:
        assert reciprocal_rank(["x", "a"], {"a"}) == 0.5
        assert reciprocal_rank(["a", "x"], {"a"}) == 1.0
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_ndcg_respects_graded_relevance(self) -> None:
        grades = {"a": 3.0, "b": 1.0}

        best_first = ndcg_at_k(["a", "b"], grades, 2)
        worst_first = ndcg_at_k(["b", "a"], grades, 2)

        assert best_first == 1.0
        assert worst_first < best_first

    def test_perfect_and_empty_rankings(self) -> None:
        assert precision_at_k([], {"a"}, 5) == 0.0
        assert recall_at_k(["a"], set(), 5) == 0.0
        assert ndcg_at_k(["a"], {}, 5) == 0.0

    def test_macro_average_weights_queries_equally(self) -> None:
        first = evaluate(["a"], {"a"}, {"a": 3.0}, ks=(1,))
        second = evaluate(["x"], {"a"}, {"a": 3.0}, ks=(1,))

        averaged = mean_metrics([first, second])

        assert averaged.precision_at_k[1] == 0.5


def _record(
    record_id: str,
    vector: tuple[float, ...],
    identity: ModelIdentity,
    *,
    paper: str = "manual:01",
) -> object:
    from researchagent.core.interfaces.vector_store import VectorMetadata, VectorRecord

    return VectorRecord(
        id=record_id,
        vector=vector,
        metadata=VectorMetadata(
            knowledge_object_id=record_id,
            paper_id=paper,
            kind=KnowledgeKind.METHOD,
            model_identity=identity,
            index_version="v1",
        ),
    )
