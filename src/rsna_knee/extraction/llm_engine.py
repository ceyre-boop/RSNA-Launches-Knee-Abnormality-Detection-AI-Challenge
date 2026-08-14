from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..constants import TARGET_LABELS
from .types import STATUSES, ExtractionResult, LabelEvidence, score_for

MODEL = "claude-sonnet-5"
PROMPT_VERSION = "v1"
# Sonnet 5 pricing per MTok (intro pricing through 2026-08-31: $2/$10)
_INPUT_COST_PER_MTOK = 2.00
_OUTPUT_COST_PER_MTOK = 10.00
_BATCH_DISCOUNT = 0.5

SYSTEM_PROMPT = """\
You extract findings from a knee MRI radiology report. The report may be in any \
language (English, Turkish, Greek, Bulgarian, Spanish, Italian, Portuguese, German, \
French, Croatian/Bosnian, Dutch, Thai, or others). Read the whole report; the \
Impression/Conclusion section is the most authoritative when it conflicts with \
Findings. Ignore the clinical question/indication section - a suspected diagnosis \
mentioned there is NOT a finding.

For each of the 12 targets, decide a status:
- "affirmed": the report states the abnormality is present.
- "negated": the report states it is absent, or describes the structure as \
normal/intact/preserved.
- "uncertain": hedged - possible, suspicious, cannot exclude, "no se puede descartar".
- "not_mentioned": the report says nothing about it.

Target definitions:
- ACL: anterior cruciate ligament abnormality (tear, sprain, degeneration, mucoid change).
- MCL: medial collateral ligament abnormality.
- Medial Meniscus: medial/internal meniscus tear or significant degeneration.
- Lateral Meniscus: lateral/external meniscus tear or significant degeneration.
- Medial OA: medial femorotibial compartment osteoarthritis or chondral loss \
(chondropathy of the medial femoral condyle or medial tibial plateau counts).
- Lateral OA: lateral femorotibial compartment osteoarthritis or chondral loss.
- PF OA: patellofemoral osteoarthritis, chondromalacia patellae, or trochlear/patellar \
chondral loss.
- Effusion: joint effusion / increased joint fluid.
- Synovitis: synovitis or synovial thickening/proliferation.
- Baker's: Baker's (popliteal) cyst.
- Contusion: bone contusion / bone marrow edema (traumatic marrow edema counts; \
degenerative subchondral edema belongs to the OA labels, not Contusion).
- Fracture: fracture (acute or subacute; do not count old healed fractures described \
as remote).

Avascular necrosis maps to NO label. Chondral/cartilage findings map to the OA label \
of their compartment. Provide a short verbatim evidence snippet from the report for \
every status that is not "not_mentioned"; use "" when not mentioned.\
"""

_LABEL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "evidence"],
    "properties": {
        "status": {"enum": list(STATUSES)},
        "evidence": {"type": "string"},
    },
}

OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["language", "labels"],
    "properties": {
        "language": {"type": "string"},
        "labels": {
            "type": "object",
            "additionalProperties": False,
            "required": list(TARGET_LABELS),
            "properties": {label: _LABEL_SCHEMA for label in TARGET_LABELS},
        },
    },
}


def estimate_cost(n_reports: int, mean_chars: float, batch: bool = True) -> float:
    report_tokens = mean_chars / 3.2
    input_tokens = n_reports * (report_tokens + 900)  # system + schema overhead
    output_tokens = n_reports * 260
    cost = (
        input_tokens / 1e6 * _INPUT_COST_PER_MTOK
        + output_tokens / 1e6 * _OUTPUT_COST_PER_MTOK
    )
    return cost * (_BATCH_DISCOUNT if batch else 1.0)


class LLMExtractor:
    engine = "llm"

    def __init__(
        self,
        cache_dir: str | Path = "data/llm_cache",
        client=None,
        model: str = MODEL,
        force: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = client
        self.model = model
        self.force = force

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    # ---- cache ----

    def _cache_path(self, study_uid: str) -> Path:
        return self.cache_dir / f"{study_uid}.json"

    def read_cache(self, study_uid: str) -> Optional[dict]:
        path = self._cache_path(study_uid)
        if not path.exists():
            return None
        try:
            entry = json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
        if entry.get("prompt_version") != PROMPT_VERSION or entry.get("model") != self.model:
            return None
        return entry

    def write_cache(self, study_uid: str, payload: dict) -> None:
        entry = {"model": self.model, "prompt_version": PROMPT_VERSION, **payload}
        self._cache_path(study_uid).write_text(json.dumps(entry, ensure_ascii=False))

    def uncached_uids(self, uids: Iterable[str]) -> List[str]:
        if self.force:
            return list(uids)
        return [u for u in uids if self.read_cache(u) is None]

    # ---- request shape ----

    def _request_params(self, report: str) -> dict:
        return {
            "model": self.model,
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT,
            "output_config": {"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
            "messages": [{"role": "user", "content": report}],
        }

    # ---- live mode ----

    def extract(self, study_uid: str, report: str) -> ExtractionResult:
        cached = None if self.force else self.read_cache(study_uid)
        if cached is None:
            cached = self._call_live(study_uid, report)
        return self.result_from_cache(study_uid, cached)

    def _call_live(self, study_uid: str, report: str) -> dict:
        try:
            response = self.client.messages.create(**self._request_params(report))
            if response.stop_reason == "refusal":
                payload = {"error": "refusal"}
            else:
                text = next(b.text for b in response.content if b.type == "text")
                payload = {"response": json.loads(text)}
        except Exception as exc:  # noqa: BLE001 — sentinel + rules fallback downstream
            payload = {"error": f"{type(exc).__name__}: {exc}"}
        self.write_cache(study_uid, payload)
        return {"model": self.model, "prompt_version": PROMPT_VERSION, **payload}

    # ---- batch mode ----

    def submit_batch(self, items: Dict[str, str]):
        """Submit uncached reports via the Message Batches API. Returns batch id."""
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        requests = [
            Request(
                custom_id=uid,
                params=MessageCreateParamsNonStreaming(**self._request_params(report)),
            )
            for uid, report in items.items()
        ]
        batch = self.client.messages.batches.create(requests=requests)
        return batch.id

    def collect_batch(self, batch_id: str) -> int:
        """Write ended-batch results to cache. Returns count written."""
        written = 0
        for result in self.client.messages.batches.results(batch_id):
            uid = result.custom_id
            if result.result.type == "succeeded":
                msg = result.result.message
                if msg.stop_reason == "refusal":
                    payload = {"error": "refusal"}
                else:
                    try:
                        text = next(b.text for b in msg.content if b.type == "text")
                        payload = {"response": json.loads(text)}
                    except (StopIteration, json.JSONDecodeError) as exc:
                        payload = {"error": f"parse: {exc}"}
            else:
                payload = {"error": f"batch:{result.result.type}"}
            self.write_cache(uid, payload)
            written += 1
        return written

    # ---- decoding ----

    def result_from_cache(self, study_uid: str, entry: dict) -> ExtractionResult:
        if "error" in entry:
            return ExtractionResult(
                study_uid=study_uid, engine=self.engine, language="unk",
                labels={}, error=str(entry["error"]),
            )
        parsed = entry["response"]
        labels: Dict[str, LabelEvidence] = {}
        for label in TARGET_LABELS:
            item = parsed["labels"][label]
            status = item["status"]
            labels[label] = LabelEvidence(
                score=score_for(status), status=status, evidence=item.get("evidence", "")
            )
        return ExtractionResult(
            study_uid=study_uid, engine=self.engine,
            language=parsed.get("language", "unk"), labels=labels,
        )
