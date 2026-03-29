"""Tests for data models and model registry."""

from artefex.models import AnalysisResult, Degradation
from artefex.models_registry import ModelRegistry, REGISTRY


def test_degradation_fields():
    d = Degradation(
        name="Test",
        confidence=0.8,
        severity=0.5,
        detail="test detail",
        category="test",
    )
    assert d.name == "Test"
    assert d.confidence == 0.8
    assert d.severity == 0.5


def test_analysis_result_empty():
    r = AnalysisResult()
    assert r.overall_severity == 0.0
    assert r.degradation_count == 0


def test_analysis_result_with_degradations():
    r = AnalysisResult(
        degradations=[
            Degradation(name="A", confidence=0.5, severity=0.3),
            Degradation(name="B", confidence=0.9, severity=0.8),
        ]
    )
    assert r.overall_severity == 0.8
    assert r.degradation_count == 2


def test_registry_list_models():
    registry = ModelRegistry()
    models = registry.list_models()
    assert len(models) == len(REGISTRY)
    assert all(m.key in REGISTRY for m in models)


def test_registry_get_model():
    registry = ModelRegistry()
    model = registry.get_model("deblock-v1")
    assert model is not None
    assert model.name == "DnCNN-3 JPEG Deblocking"


def test_registry_get_unknown_model():
    registry = ModelRegistry()
    model = registry.get_model("nonexistent")
    assert model is None


def test_registry_get_model_for_category():
    registry = ModelRegistry()
    model = registry.get_model_for_category("compression")
    assert model is not None
    assert model.category == "compression"


def test_plugin_registry_loads():
    from artefex.plugins import get_plugin_registry
    registry = get_plugin_registry()
    info = registry.list_plugins()
    assert "detectors" in info
    assert "restorers" in info
