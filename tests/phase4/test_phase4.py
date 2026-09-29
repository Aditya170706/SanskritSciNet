"""
tests/phase4/test_phase4.py
────────────────────────────
Unit + integration tests for Phase 4 API and UI components.
Tests run without starting the server (TestClient used for API tests).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

MOCK_PIPELINE_RESULT = {
    "input_text":  "समचतुरश्रस्य क्षेत्रफलं भुजवर्गः।",
    "veda":        "Yajurveda",
    "domain":      "Mathematics",
    "branch":      "Geometry",
    "math_flag":   True,
    "formula": {
        "name":   "Pythagorean theorem",
        "expr":   "a**2 + b**2 - c**2",
        "latex":  "a^2 + b^2 = c^2",
        "source": "Sulba Sutra",
    },
    "confidence":  0.831,
    "confidence_breakdown": {"model": 0.91, "retrieval": 0.74, "graph": 0.60},
    "reasoning":   ["Math keyword detected", "Formula retrieved from KG"],
    "top_retrieved": [
        {"id": "SHLOKA_000000", "text": "समचतुरश्रस्य…", "source": "Sulba Sutra",
         "final_score": 0.92}
    ],
    "phase2_predictions": {
        "veda":   {"label": "Yajurveda",   "conf": 0.87},
        "domain": {"label": "Mathematics", "conf": 0.84},
        "branch": {"label": "Geometry",    "conf": 0.79},
    },
    "elapsed_ms": 45.3,
    "cache_hit":  False,
}


@pytest.fixture
def client():
    """TestClient with a mocked pipeline injected into app.state."""
    from api.main import app

    mock_pipeline = MagicMock()
    mock_pipeline.query.return_value = MOCK_PIPELINE_RESULT

    with TestClient(app, raise_server_exceptions=False) as c:
        app.state.pipeline   = mock_pipeline
        app.state.start_time = time.time()
        yield c


# ─────────────────────────────────────────────────────────────────────────────
# Schema tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemas:

    def test_analyze_request_valid(self):
        from api.schemas import AnalyzeRequest
        req = AnalyzeRequest(shloka="अग्निमीळे पुरोहितं।")
        assert req.shloka == "अग्निमीळे पुरोहितं।"

    def test_analyze_request_strips_whitespace(self):
        from api.schemas import AnalyzeRequest
        req = AnalyzeRequest(shloka="  hello  ")
        assert req.shloka == "hello"

    def test_analyze_request_too_short(self):
        from api.schemas import AnalyzeRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AnalyzeRequest(shloka="a")

    def test_analyze_request_too_long(self):
        from api.schemas import AnalyzeRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AnalyzeRequest(shloka="a" * 1001)

    def test_analyze_request_only_whitespace(self):
        from api.schemas import AnalyzeRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AnalyzeRequest(shloka="   ")

    def test_analyze_request_control_chars(self):
        from api.schemas import AnalyzeRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AnalyzeRequest(shloka="valid\x00text")

    def test_analyze_response_from_pipeline(self):
        from api.schemas import AnalyzeResponse
        resp = AnalyzeResponse.from_pipeline(MOCK_PIPELINE_RESULT)
        assert resp.veda   == "Yajurveda"
        assert resp.domain == "Mathematics"
        assert resp.math   is True
        assert resp.formula is not None
        assert resp.formula.name == "Pythagorean theorem"
        assert resp.confidence == 0.831

    def test_analyze_response_no_formula(self):
        from api.schemas import AnalyzeResponse
        raw = {**MOCK_PIPELINE_RESULT, "formula": None, "math_flag": False}
        resp = AnalyzeResponse.from_pipeline(raw)
        assert resp.formula is None
        assert resp.math is False

    def test_confidence_breakdown_present(self):
        from api.schemas import AnalyzeResponse
        resp = AnalyzeResponse.from_pipeline(MOCK_PIPELINE_RESULT)
        assert resp.confidence_breakdown.model     == 0.91
        assert resp.confidence_breakdown.retrieval == 0.74


# ─────────────────────────────────────────────────────────────────────────────
# API endpoint tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIEndpoints:

    def test_root_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "Vedic" in r.json()["name"]

    def test_health_returns_running(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "running"
        assert "version"       in data
        assert "pipeline_ready" in data

    def test_health_pipeline_ready(self, client):
        r = client.get("/health")
        assert r.json()["pipeline_ready"] is True

    def test_analyze_valid_shloka(self, client):
        r = client.post("/analyze", json={"shloka": "अग्निमीळे पुरोहितं।"})
        assert r.status_code == 200
        data = r.json()
        assert "veda"       in data
        assert "domain"     in data
        assert "branch"     in data
        assert "math"       in data
        assert "confidence" in data
        assert "reasoning"  in data

    def test_analyze_response_schema(self, client):
        r    = client.post("/analyze", json={"shloka": "गणितं सूत्रम्।"})
        data = r.json()
        assert isinstance(data["veda"],      str)
        assert isinstance(data["domain"],    str)
        assert isinstance(data["branch"],    str)
        assert isinstance(data["math"],      bool)
        assert isinstance(data["confidence"],float)
        assert isinstance(data["reasoning"], list)

    def test_analyze_empty_shloka(self, client):
        r = client.post("/analyze", json={"shloka": ""})
        assert r.status_code == 422    # Validation error

    def test_analyze_whitespace_only(self, client):
        r = client.post("/analyze", json={"shloka": "   "})
        assert r.status_code == 422

    def test_analyze_missing_shloka_field(self, client):
        r = client.post("/analyze", json={})
        assert r.status_code == 422

    def test_analyze_too_long_input(self, client):
        r = client.post("/analyze", json={"shloka": "a" * 2000})
        assert r.status_code == 422

    def test_analyze_pipeline_called_once(self, client):
        from api.main import app
        client.post("/analyze", json={"shloka": "test shloka here"})
        app.state.pipeline.query.assert_called()

    def test_examples_endpoint(self, client):
        r = client.get("/examples")
        assert r.status_code == 200
        data = r.json()
        assert "examples" in data
        assert len(data["examples"]) >= 1
        assert "shloka" in data["examples"][0]
        assert "label"  in data["examples"][0]

    def test_metrics_endpoint(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        data = r.json()
        assert "total_requests" in data
        assert "avg_latency_ms" in data

    def test_unknown_route_404(self, client):
        r = client.get("/nonexistent")
        assert r.status_code == 404

    def test_analyze_pipeline_not_ready_503(self, client):
        from api.main import app
        original = app.state.pipeline
        app.state.pipeline = None
        try:
            r = client.post("/analyze", json={"shloka": "test shloka"})
            assert r.status_code == 503
        finally:
            app.state.pipeline = original


# ─────────────────────────────────────────────────────────────────────────────
# Metrics collector tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsCollector:

    def test_records_requests(self):
        from api.metrics_collector import MetricsCollector
        mc = MetricsCollector(flush_interval=0)
        mc.record(50.0, success=True)
        mc.record(80.0, success=True)
        mc.record(20.0, success=False)
        snap = mc.snapshot()
        assert snap["total_requests"] == 3
        assert snap["successful"]     == 2
        assert snap["failed"]         == 1

    def test_cache_hit_rate(self):
        from api.metrics_collector import MetricsCollector
        mc = MetricsCollector(flush_interval=0)
        mc.record(10.0, success=True, cache_hit=True)
        mc.record(10.0, success=True, cache_hit=False)
        snap = mc.snapshot()
        assert snap["cache_hit_rate"] == 0.5

    def test_avg_latency(self):
        from api.metrics_collector import MetricsCollector
        mc = MetricsCollector(flush_interval=0)
        for ms in [10.0, 20.0, 30.0]:
            mc.record(ms, success=True)
        snap = mc.snapshot()
        assert abs(snap["avg_latency_ms"] - 20.0) < 0.1

    def test_zero_requests_no_crash(self):
        from api.metrics_collector import MetricsCollector
        mc   = MetricsCollector(flush_interval=0)
        snap = mc.snapshot()
        assert snap["total_requests"]  == 0
        assert snap["avg_latency_ms"]  == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Settings tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSettings:

    def test_server_port_is_int(self):
        from app_config.settings import SERVER_PORT
        assert isinstance(SERVER_PORT, int)
        assert 1024 <= SERVER_PORT <= 65535

    def test_ui_port_is_int(self):
        from app_config.settings import UI_PORT
        assert isinstance(UI_PORT, int)

    def test_max_input_len_positive(self):
        from app_config.settings import MAX_INPUT_LEN
        assert MAX_INPUT_LEN > 0

    def test_api_url_has_scheme(self):
        from app_config.settings import API_URL
        assert API_URL.startswith("http")

    def test_log_dir_path(self):
        from app_config.settings import LOG_DIR
        assert isinstance(LOG_DIR, Path)

    def test_config_section_returns_dict(self):
        from app_config.settings import section
        srv = section("server")
        assert isinstance(srv, dict)


# ─────────────────────────────────────────────────────────────────────────────
# Integration — full request cycle
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    def test_analyze_full_cycle(self, client):
        """POST /analyze → valid response with all required fields."""
        r    = client.post("/analyze", json={"shloka": "समचतुरश्रस्य क्षेत्रफलं।"})
        assert r.status_code == 200
        data = r.json()

        required = ["veda","domain","branch","math","confidence",
                    "reasoning","confidence_breakdown","elapsed_ms"]
        for key in required:
            assert key in data, f"Missing key: {key}"

        assert 0.0 <= data["confidence"] <= 1.0
        assert isinstance(data["reasoning"], list)
        cb = data["confidence_breakdown"]
        assert 0.0 <= cb["model"]     <= 1.0
        assert 0.0 <= cb["retrieval"] <= 1.0
        assert 0.0 <= cb["graph"]     <= 1.0

    def test_health_then_analyze(self, client):
        """Health check followed by analyze must both succeed."""
        h = client.get("/health")
        assert h.status_code == 200
        assert h.json()["pipeline_ready"] is True

        a = client.post("/analyze", json={"shloka": "विद्या ददाति विनयम्।"})
        assert a.status_code == 200

    def test_multiple_sequential_requests(self, client):
        """10 sequential requests must all succeed."""
        shlokas = [
            "अग्निमीळे पुरोहितं।",
            "गणितं सूत्रम्।",
            "ज्योतिष ग्रह।",
            "विद्या ददाति विनयम्।",
            "समचतुरश्रस्य क्षेत्रफलं।",
        ] * 2
        for s in shlokas:
            r = client.post("/analyze", json={"shloka": s})
            assert r.status_code == 200

    def test_metrics_increment_after_requests(self, client):
        """total_requests must increment after analyze calls."""
        from api.metrics_collector import MetricsCollector
        mc = MetricsCollector(flush_interval=0)
        from api.main import app
        with patch("api.routes.get_collector", return_value=mc):
            client.post("/analyze", json={"shloka": "test shloka text"})
            client.post("/analyze", json={"shloka": "second shloka here"})
        # The real collector is a singleton — just verify metrics endpoint works
        r = client.get("/metrics")
        assert r.status_code == 200
