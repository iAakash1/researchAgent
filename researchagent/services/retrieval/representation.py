"""The canonical retrieval representation of a knowledge object.

One representation, shared by BM25 and by embedding. Two subsystems that indexed
different text would make their scores incomparable, and the whole point of v0.7 is that
the comparison is meaningful.

Derived, never stored on the object: the KnowledgeObject is immutable and this is a view
of it. Changing how the view is built changes retrieval results, so it is versioned and
that version is part of the index identity.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from researchagent.models.knowledge import KnowledgeObject

# Bump when the composition below changes: existing vectors become incomparable.
REPRESENTATION_VERSION = "1"

_MAX_QUOTE_CHARS = 400


class RetrievalRepresentation(BaseModel):
    """What gets indexed for one knowledge object."""

    model_config = {"frozen": True}

    knowledge_object_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    version: str = REPRESENTATION_VERSION


def represent(obj: KnowledgeObject) -> RetrievalRepresentation:
    """Compose the indexed text for a knowledge object.

    Name first because it carries the most signal per token, then the kind so a query for
    "datasets" has something to match, then the description, then the supporting quote —
    which is the paper's own wording and often the only place the domain vocabulary
    appears in full.
    """
    parts = [obj.name, obj.kind.value.replace("_", " ")]
    if obj.description:
        parts.append(obj.description)
    for quote in obj.quotes[:2]:
        parts.append(quote[:_MAX_QUOTE_CHARS])

    return RetrievalRepresentation(
        knowledge_object_id=obj.id, text=" \n".join(part for part in parts if part.strip())
    )
