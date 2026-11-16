import json
from types import SimpleNamespace

import pytest

from app.knowledge import KnowledgeBase, load_chunks
from app.references import GENRE_TARGETS, compare
from app.tips import ClaudeTipWriter, LLMError, diagnose, generate_tips


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


def base_metrics(**overrides):
    m = {
        "sr": 44100, "duration_s": 180.0, "bpm": 124.0, "key": "A", "mode": "minor",
        "key_confidence": 0.7, "lufs_i": -10.0, "lra": 5.0, "true_peak_dbfs": -1.2,
        "sample_peak_dbfs": -1.4, "plr_db": 8.8, "clipped_samples": 0,
        "crest_factor_db": 9.0, "rms_spread_db": 6.0,
        "band_level_db": {"sub": -3.0, "low": -5.0, "low_mid": -10.0, "high_mid": -16.0, "high": -18.0},
        "spectral_centroid_mean_hz": 2500.0, "spectral_rolloff_hz": 6000.0,
        "channels": 2, "stereo_width": 0.18, "lr_correlation": 0.8, "low_end_width": 0.02,
    }
    m.update(overrides)
    return m


def ids(findings):
    return {f.id for f in findings}


def test_knowledge_chunks_parse():
    chunks = load_chunks()
    assert len(chunks) >= 20
    assert len({c.id for c in chunks}) == len(chunks), "chunk ids must be unique"
    assert all(c.tags and c.text for c in chunks)


@pytest.mark.parametrize("query, expected", [
    ("true peak inter-sample clipping ceiling lossy encoding", "true-peak-and-inter-sample-clipping"),
    ("muddy low mids boxy congested", "muddy-low-mids"),
    ("low end width bass mono stereo bass sub side", "keep-the-low-end-mono"),
    ("dull dark top end air high frequencies brightness", "dull-or-too-bright-top-end"),
    ("quiet master too quiet raise loudness limiter clipper", "raising-loudness-without-crushing-the-mix"),
])
def test_retrieval_finds_the_right_chunk(kb, query, expected):
    assert kb.search(query, k=1)[0][0].id == expected


def test_clean_mix_has_no_serious_findings():
    findings = diagnose(base_metrics())
    assert not [f for f in findings if f.severity in ("high", "medium")]


def test_absolute_problems_are_flagged():
    m = base_metrics(true_peak_dbfs=0.4, clipped_samples=500, plr_db=5.0,
                     lr_correlation=-0.2, low_end_width=0.3)
    found = ids(diagnose(m))
    assert {"true_peak_over", "clipping", "over_limited", "phase_negative", "stereo_bass"} <= found
    assert diagnose(m)[0].severity == "high"


def test_genre_target_comparison_findings():
    found = ids(diagnose(base_metrics(lufs_i=-14.0, plr_db=12.8), GENRE_TARGETS["pop"]["metrics"], "pop"))
    assert {"quieter_than_ref", "more_dynamic"} <= found


def test_uploaded_reference_tonal_findings():
    ref = base_metrics()
    mix = base_metrics(spectral_centroid_mean_hz=1800.0,
                       band_level_db={"sub": -9.0, "low": -4.0, "low_mid": -6.0, "high_mid": -16.5, "high": -18.0})
    found = ids(diagnose(mix, ref, "my reference"))
    assert {"darker_than_ref", "band_low_mid_high", "band_sub_low"} <= found
    assert not {"band_low_high", "band_high_mid_low", "band_high_low"} & found


def test_compare_rows_and_tolerance():
    rows = compare(base_metrics(lufs_i=-9.5), GENRE_TARGETS["pop"]["metrics"])
    lufs = next(r for r in rows if r["metric"] == "lufs_i")
    assert lufs["delta"] == pytest.approx(-0.5)
    assert lufs["within_tolerance"]
    assert not any(r["metric"].startswith("band_level_db.") for r in rows)  # genre targets have no bands
    tonal = compare(base_metrics(), base_metrics())
    assert any(r["metric"] == "band_level_db.sub" and r["within_tolerance"] for r in tonal)


def test_rule_engine_output(kb):
    result = generate_tips(base_metrics(true_peak_dbfs=0.2), kb)
    assert result["engine"] == "rules"
    assert result["tips"][0]["title"] == "True peak at or above 0 dBTP"
    source_ids = {s["id"] for s in result["sources"]}
    for tip in result["tips"]:
        assert set(tip["sources"]) <= source_ids


class FakeClient:
    """Stands in for anthropic.Anthropic; records the request."""

    def __init__(self, response):
        self.calls = []
        outer = self

        class _Messages:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                return response

        self.beta = SimpleNamespace(messages=_Messages())


def fake_response(payload, stop_reason="end_turn"):
    return SimpleNamespace(stop_reason=stop_reason,
                           content=[SimpleNamespace(type="thinking", thinking=""),
                                    SimpleNamespace(type="text", text=json.dumps(payload))])


def test_claude_engine_uses_retrieved_sources(kb):
    payload = {"summary": "Peaks too hot.", "tips": [{
        "title": "Lower the ceiling", "priority": "high", "category": "loudness",
        "detail": "True peak is +0.2 dBTP.", "steps": ["Set ceiling to -1 dBTP"],
        "sources": ["true-peak-and-inter-sample-clipping", "made-up-id"]}]}
    client = FakeClient(fake_response(payload))
    writer = ClaudeTipWriter(model="claude-opus-5-5", client=client)
    result = generate_tips(base_metrics(true_peak_dbfs=0.2), kb, writer=writer)

    assert result["engine"] == "claude"
    assert result["tips"][0]["sources"] == ["true-peak-and-inter-sample-clipping"]
    request = client.calls[0]
    assert request["model"] == "claude-opus-5-5"
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert "true-peak-and-inter-sample-clipping" in request["messages"][0]["content"]


def test_claude_refusal_falls_back_to_rules(kb):
    writer = ClaudeTipWriter(client=FakeClient(fake_response({}, stop_reason="refusal")))
    result = generate_tips(base_metrics(true_peak_dbfs=0.2), kb, writer=writer)
    assert result["engine"] == "rules"
    assert "declined" in result["llm_error"]
    assert result["tips"]


def test_claude_invalid_json_raises():
    bad = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="nope")])
    with pytest.raises(LLMError):
        ClaudeTipWriter(client=FakeClient(bad)).write("x", set())
