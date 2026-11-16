"""Diagnose a mix from its metrics, retrieve relevant knowledge, write tips.

Two engines produce the tips:
- "claude": Claude writes them from the metrics, findings and retrieved
  knowledge (needs ANTHROPIC_API_KEY).
- "rules": deterministic tips straight from the findings. Used when no key is
  configured, or as a fallback when the API call fails.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass

from .knowledge import Chunk, KnowledgeBase

log = logging.getLogger(__name__)

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


@dataclass
class Finding:
    id: str
    category: str      # loudness | dynamics | tone | stereo | workflow
    severity: str      # high | medium | low | info
    title: str
    observation: str   # what the numbers say
    action: str        # what to do about it
    query: str         # retrieval query for the knowledge base


def diagnose(metrics: dict, reference: dict | None = None, reference_name: str = "the reference") -> list[Finding]:
    """Turn metrics (and an optional reference profile) into findings."""
    f: list[Finding] = []

    def add(*fields):
        f.append(Finding(*fields))

    tp, lufs, plr = metrics["true_peak_dbfs"], metrics["lufs_i"], metrics["plr_db"]

    # --- Absolute checks -------------------------------------------------
    if metrics.get("clipped_samples", 0) > 10:
        add("clipping", "loudness", "high", "Clipped samples in the file",
            f"{metrics['clipped_samples']} samples sit at full scale.",
            "Find the stage that clips (master bus, post-limiter plug-ins, render settings) and render with the limiter ceiling below 0 dBFS.",
            "digital clipping clipped samples 0 dbfs distortion")
    if tp > -0.1:
        add("true_peak_over", "loudness", "high", "True peak at or above 0 dBTP",
            f"True peak is {tp:+.1f} dBTP, so it will distort after MP3/AAC encoding.",
            "Enable true-peak detection on the final limiter and set the ceiling to -1 dBTP (-1.5 to -2 for very loud masters).",
            "true peak inter-sample clipping ceiling lossy encoding")
    elif tp > -1.0:
        add("true_peak_high", "loudness", "medium", "True peak above -1 dBTP",
            f"True peak is {tp:.1f} dBTP; streaming services recommend -1 dBTP or lower.",
            "Lower the limiter ceiling to -1 dBTP with oversampling / true-peak mode on.",
            "true peak dbtp ceiling inter-sample lossy encoding")
    if plr < 7:
        add("over_limited", "dynamics", "high" if plr < 5.5 else "medium", "Master looks over-limited",
            f"Peak-to-loudness ratio is {plr:.1f} dB (crest factor {metrics.get('crest_factor_db', 0):.1f} dB): transients are heavily squashed.",
            "Back the limiter off to 3-4 dB of peak gain reduction and get loudness from gentler stages (bus compression, soft clipping).",
            "over-limited plr low crest factor squashed limiter punch transients")
    if metrics.get("channels") == 2:
        corr, width = metrics["lr_correlation"], metrics["stereo_width"]
        if corr < 0:
            add("phase_negative", "stereo", "high", "Out-of-phase stereo content",
                f"L/R correlation is {corr:.2f}; parts of the mix will cancel in mono.",
                "Check polarity of multi-mic sources and pull back stereo wideners; check the mix in mono.",
                "phase negative correlation mono compatibility cancellation stereo widener")
        elif corr < 0.3:
            add("phase_low", "stereo", "medium", "Weak mono compatibility",
                f"L/R correlation is only {corr:.2f}.",
                "Audition in mono and reduce side-channel processing on the elements that disappear.",
                "mono compatibility correlation too wide phase")
        if metrics.get("low_end_width", 0) > 0.15:
            add("stereo_bass", "stereo", "medium", "Stereo energy in the low end",
                f"{metrics['low_end_width']*100:.0f}% of the energy below 150 Hz is in the side channel.",
                "Make everything below 120-150 Hz mono (bass-mono tool or a high-pass on the side channel).",
                "low end width bass mono stereo bass sub side")
        if width < 0.01:
            add("near_mono", "stereo", "low", "The mix is almost mono",
                "Left and right channels are nearly identical.",
                "Pan supporting parts, double-track, and use stereo ambience to create width.",
                "narrow mono stereo width panning")
        elif width > 0.4:
            add("very_wide", "stereo", "medium", "Very wide stereo image",
                f"{width*100:.0f}% of the energy is in the side channel.",
                "Check mono compatibility and reduce stereo widening on the busiest elements.",
                "too wide stereo widener side mono compatibility")
    else:
        add("mono_file", "stereo", "info", "Mono file",
            "The file has a single channel, so no stereo analysis was possible.",
            "If this is a stereo mix, export it as a stereo file.",
            "narrow mono stereo width")
    if metrics.get("duration_s", 0) > 60 and metrics["lra"] < 3:
        add("flat_macro", "dynamics", "low", "Little loudness contrast between sections",
            f"Loudness range is {metrics['lra']:.1f} LU.",
            "Automate whole sections (+1 to +2 dB into choruses) and thin out verses to create contrast.",
            "loudness range lra macro dynamics sections automation")

    # --- Relative to a reference ------------------------------------------
    if reference:
        ref = reference
        if "lufs_i" in ref:
            d = lufs - ref["lufs_i"]
            if d < -2:
                add("quieter_than_ref", "loudness", "medium", f"Quieter than {reference_name}",
                    f"{lufs:.1f} LUFS vs {ref['lufs_i']:.1f} LUFS ({d:+.1f} LU).",
                    "Spread the extra gain across several gentle stages instead of one limiter.",
                    "quiet master too quiet raise loudness limiter clipper")
            elif d > 2:
                add("louder_than_ref", "loudness", "medium", f"Louder than {reference_name}",
                    f"{lufs:.1f} LUFS vs {ref['lufs_i']:.1f} LUFS ({d:+.1f} LU).",
                    "Ease off the limiter; streaming normalisation will turn the track down anyway.",
                    "master too loud over-limited streaming normalisation")
        if "plr_db" in ref and plr >= 7:
            d = plr - ref["plr_db"]
            if d < -3:
                add("more_compressed", "dynamics", "medium", f"More compressed than {reference_name}",
                    f"PLR {plr:.1f} dB vs {ref['plr_db']:.1f} dB.",
                    "Reduce master-bus compression/limiting; use parallel compression for density.",
                    "compression punch transients crest factor squashed")
            elif d > 4:
                add("more_dynamic", "dynamics", "low", f"More dynamic than {reference_name}",
                    f"PLR {plr:.1f} dB vs {ref['plr_db']:.1f} dB.",
                    "Tame the loudest peaks (drum bus clipping, vocal automation) before the limiter.",
                    "too dynamic plr high peaks inconsistent")
        if "lra" in ref and metrics.get("duration_s", 0) > 30:
            d = metrics["lra"] - ref["lra"]
            if d < -3 or d > 4:
                add("lra_diff", "dynamics", "low", "Different loudness contrast between sections",
                    f"LRA {metrics['lra']:.1f} LU vs {ref['lra']:.1f} LU in {reference_name}.",
                    "Adjust section-level automation and arrangement density to match the intended contrast.",
                    "loudness range lra macro dynamics sections")
        if "stereo_width" in ref and metrics.get("channels") == 2:
            d = metrics["stereo_width"] - ref["stereo_width"]
            if d < -0.08:
                add("narrower_than_ref", "stereo", "low", f"Narrower than {reference_name}",
                    f"Side energy {metrics['stereo_width']*100:.0f}% vs {ref['stereo_width']*100:.0f}%.",
                    "Widen supporting parts with panning, doubles and stereo reverbs.",
                    "narrow stereo width panning image")
            elif d > 0.12:
                add("wider_than_ref", "stereo", "low", f"Wider than {reference_name}",
                    f"Side energy {metrics['stereo_width']*100:.0f}% vs {ref['stereo_width']*100:.0f}%.",
                    "Make sure the width doesn't cost mono compatibility; keep the centre strong.",
                    "too wide mono compatibility side")
        if "spectral_centroid_mean_hz" in ref:
            d = metrics["spectral_centroid_mean_hz"] - ref["spectral_centroid_mean_hz"]
            if d < -400:
                add("darker_than_ref", "tone", "medium", f"Darker than {reference_name}",
                    f"Spectral centroid {metrics['spectral_centroid_mean_hz']:.0f} Hz vs {ref['spectral_centroid_mean_hz']:.0f} Hz.",
                    "Add air with a broad high shelf or saturation on the elements that should shine.",
                    "dull dark top end air high frequencies brightness")
            elif d > 400:
                add("brighter_than_ref", "tone", "medium", f"Brighter than {reference_name}",
                    f"Spectral centroid {metrics['spectral_centroid_mean_hz']:.0f} Hz vs {ref['spectral_centroid_mean_hz']:.0f} Hz.",
                    "Check hats/cymbals level and harsh resonances; a gentle high shelf cut on the bus may help.",
                    "too bright harsh treble high frequencies brightness")
        ref_bands = ref.get("band_level_db") or {}
        mix_bands = metrics.get("band_level_db") or {}
        band_rules = {
            "sub": ("sub bass", "too much sub bass boomy rumble headroom", "not enough sub thin weak low end"),
            "low": ("low end (60-250 Hz)", "too much low end boomy kick bass headroom", "thin weak low end lacks bass kick bass"),
            "low_mid": ("low mids (250 Hz-2 kHz)", "muddy low mids boxy congested", "thin low mids lacks body"),
            "high_mid": ("upper mids (2-6 kHz)", "harsh upper mids presence fatigue", "lacks presence upper mids dull"),
            "high": ("highs (6 kHz+)", "too bright treble air harsh", "dull dark top end air"),
        }
        for band, (label, too_much_q, too_little_q) in band_rules.items():
            if band not in ref_bands or band not in mix_bands:
                continue
            d = float(mix_bands[band]) - float(ref_bands[band])
            if abs(d) > 3:
                more = d > 0
                add(f"band_{band}_{'high' if more else 'low'}", "tone", "medium" if abs(d) > 6 else "low",
                    f"{'More' if more else 'Less'} {label} than {reference_name}",
                    f"Relative to the rest of the spectrum this band sits {d:+.1f} dB vs the reference.",
                    ("Find which elements over-feed this range and cut them there, not on the master."
                     if more else "Strengthen this range on the elements that should own it."),
                    too_much_q if more else too_little_q)
    else:
        if lufs > -9:
            add("loud_for_streaming", "loudness", "info", "Loud for streaming platforms",
                f"At {lufs:.1f} LUFS the track will be turned down by about {lufs + 14:.0f} dB on Spotify/YouTube.",
                "That's fine if the loudness is a stylistic choice; otherwise, keep more transients.",
                "loudness targets streaming normalisation spotify lufs")
        elif lufs < -18:
            add("quiet_master", "loudness", "info", "Quiet master",
                f"At {lufs:.1f} LUFS the track is below every streaming target.",
                "If this is a finished master, raise loudness gradually across several stages.",
                "quiet master too quiet raise loudness limiter")

    f.sort(key=lambda x: SEVERITY_ORDER[x.severity])
    return f


# --- Tip engines ------------------------------------------------------------

def rule_based_tips(findings: list[Finding], kb: KnowledgeBase) -> dict:
    tips = []
    for fd in findings:
        hits = kb.search(fd.query, k=1)
        tips.append({
            "title": fd.title,
            "priority": fd.severity,
            "category": fd.category,
            "detail": f"{fd.observation} {fd.action}",
            "steps": [],
            "sources": [hits[0][0].id] if hits else [],
        })
    serious = [fd for fd in findings if fd.severity in ("high", "medium")]
    if not findings:
        summary = "No problems stood out in the measurements. Trust your ears and check translation on several systems."
    elif not serious:
        summary = f"The mix measures well; {len(findings)} minor point(s) to consider."
    else:
        summary = f"{len(serious)} issue(s) worth fixing first: " + "; ".join(fd.title for fd in serious[:3]) + "."
    return {"summary": summary, "tips": tips}


TIPS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "tips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "priority": {"type": "string", "enum": ["high", "medium", "low", "info"]},
                    "category": {"type": "string", "enum": ["loudness", "dynamics", "tone", "stereo", "workflow"]},
                    "detail": {"type": "string"},
                    "steps": {"type": "array", "items": {"type": "string"}},
                    "sources": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "priority", "category", "detail", "steps", "sources"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "tips"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are Producer's Copilot, an experienced mixing and mastering engineer \
reviewing a producer's track from objective measurements.

You receive: the track's DSP metrics, an optional comparison with a reference \
(a commercial track the producer uploaded, or a genre profile), findings that \
automatic rules flagged, and excerpts from a production knowledge base, each \
with an id.

Write 3 to 7 tips, most important first. Each tip should:
- name the problem in plain words and cite the numbers that show it;
- give concrete actions with typical settings (frequencies, ratios, dB);
- list in `sources` the ids of the knowledge excerpts it relies on (only ids \
you were given; an empty list if none apply).

Measurements are hints, not verdicts: a genre profile is approximate, and \
automatic key and tempo detection can be wrong (relative major/minor, double \
or half tempo). Say so when it matters. Don't invent measurements you were \
not given, and don't flag things that measure fine. If the track measures \
well, say so and keep the list short.

`summary` is two sentences at most: the overall state of the mix and the \
first thing to fix."""


def _round(value):
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    return value


def build_user_prompt(metrics: dict, comparison: list[dict], reference_name: str | None,
                      findings: list[Finding], chunks: list[Chunk]) -> str:
    parts = ["<metrics>", json.dumps(_round(metrics), indent=1), "</metrics>"]
    if comparison:
        parts += [f"<comparison reference=\"{reference_name}\">",
                  json.dumps([_round(r) for r in comparison], indent=1), "</comparison>"]
    else:
        parts.append("<comparison>No reference selected.</comparison>")
    parts += ["<findings>", json.dumps([asdict(fd) for fd in findings], indent=1), "</findings>",
              "<knowledge>"]
    for c in chunks:
        parts.append(f"<excerpt id=\"{c.id}\" title=\"{c.title}\">\n{c.text}\n</excerpt>")
    parts.append("</knowledge>")
    return "\n".join(parts)


class LLMError(RuntimeError):
    pass


class ClaudeTipWriter:
    """Writes tips with Claude. Create only when an API key is configured."""

    def __init__(self, model: str | None = None, client=None):
        import anthropic  # imported lazily so the rules engine has no SDK dependency at runtime

        self.model = model or os.environ.get("COPILOT_MODEL", "claude-opus-5-5")
        self.client = client or anthropic.Anthropic(timeout=120.0, max_retries=2)
        self._anthropic = anthropic

    def write(self, user_prompt: str, allowed_sources: set[str]) -> dict:
        anthropic = self._anthropic
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                output_config={"effort": "medium",
                               "format": {"type": "json_schema", "schema": TIPS_SCHEMA}},
                # Server-side fallback: if the model declines, the API retries on
                # a fallback model inside the same call.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Could not reach the Claude API: {exc}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Claude API rate limit reached") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error {exc.status_code}: {exc.message}") from exc

        if response.stop_reason == "refusal":
            raise LLMError("The model declined to answer")
        if response.stop_reason == "max_tokens":
            raise LLMError("The response was cut off")
        text = "".join(b.text for b in response.content if b.type == "text")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError("The model returned invalid JSON") from exc
        for tip in data.get("tips", []):
            tip["sources"] = [s for s in tip.get("sources", []) if s in allowed_sources]
        return data


def generate_tips(metrics: dict, kb: KnowledgeBase, *, reference: dict | None = None,
                  reference_name: str | None = None, comparison: list[dict] | None = None,
                  writer: ClaudeTipWriter | None = None) -> dict:
    """Full pipeline: diagnose -> retrieve -> write (Claude or rules)."""
    findings = diagnose(metrics, reference, reference_name or "the reference")
    queries = [fd.query for fd in findings] or ["reference tracks workflow mix translation checks"]
    chunks = kb.retrieve_for(queries)
    result = {
        "findings": [asdict(fd) for fd in findings],
        "sources": [c.as_dict() for c in chunks],
        "engine": "rules",
        "llm_error": None,
    }
    if writer is not None:
        prompt = build_user_prompt(metrics, comparison or [], reference_name, findings, chunks)
        try:
            written = writer.write(prompt, {c.id for c in chunks})
            result.update(written, engine="claude")
            return result
        except LLMError as exc:
            log.warning("Falling back to rule-based tips: %s", exc)
            result["llm_error"] = str(exc)
    rules = rule_based_tips(findings, kb)
    # Rule tips may cite a chunk outside the retrieved set; include it
    known = {c["id"] for c in result["sources"]}
    for tip in rules["tips"]:
        for sid in tip["sources"]:
            if sid not in known and kb.get(sid):
                result["sources"].append(kb.get(sid).as_dict())
                known.add(sid)
    result.update(rules)
    return result
