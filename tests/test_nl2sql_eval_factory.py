from app.nl2sql_eval.factory import required_namespaces_ready
from app.rag.models import RagSourceType


class FakeVectorStore:
    def __init__(self, namespaces):
        self.namespaces = set(namespaces)

    def has_namespace(self, source_type, namespace):
        return (source_type, namespace) in self.namespaces


def test_factory_requires_every_source_namespace_for_index_reuse():
    namespace = "shop"
    ddl_only = FakeVectorStore({(RagSourceType.DDL, namespace)})
    sample_only = FakeVectorStore({(RagSourceType.SAMPLE_VALUE, namespace)})
    both = FakeVectorStore({(RagSourceType.DDL, namespace), (RagSourceType.SAMPLE_VALUE, namespace)})

    assert required_namespaces_ready(both, namespace) is True
    assert required_namespaces_ready(ddl_only, namespace) is False
    assert required_namespaces_ready(ddl_only, namespace, (RagSourceType.DDL,)) is True
    assert required_namespaces_ready(sample_only, namespace, (RagSourceType.SAMPLE_VALUE,)) is True
    assert required_namespaces_ready(sample_only, namespace, (RagSourceType.DDL,)) is False
