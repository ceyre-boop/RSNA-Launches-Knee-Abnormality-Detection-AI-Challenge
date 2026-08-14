import json
from types import SimpleNamespace

import pytest

from rsna_knee.constants import TARGET_LABELS
from rsna_knee.extraction.llm_engine import (
    OUTPUT_SCHEMA,
    PROMPT_VERSION,
    LLMExtractor,
    estimate_cost,
)


def make_response(status_map=None, stop_reason="end_turn"):
    labels = {
        label: {"status": (status_map or {}).get(label, "not_mentioned"), "evidence": ""}
        for label in TARGET_LABELS
    }
    text = json.dumps({"language": "en", "labels": labels})
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
    )


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def make_client(response):
    return SimpleNamespace(messages=FakeMessages(response))


def test_valid_response_and_cache(tmp_path):
    client = make_client(make_response({"ACL": "affirmed", "Effusion": "negated"}))
    llm = LLMExtractor(cache_dir=tmp_path, client=client)
    result = llm.extract("uid1", "ACL tear, no effusion.")
    assert result.labels["ACL"].status == "affirmed"
    assert result.labels["Effusion"].score < 0.1
    assert client.messages.calls == 1

    # Cache hit: client must not be called again
    client2 = make_client(AssertionError("must not call API"))
    llm2 = LLMExtractor(cache_dir=tmp_path, client=client2)
    result2 = llm2.extract("uid1", "ACL tear, no effusion.")
    assert result2.labels["ACL"].status == "affirmed"
    assert client2.messages.calls == 0


def test_force_recalls(tmp_path):
    client = make_client(make_response())
    llm = LLMExtractor(cache_dir=tmp_path, client=client)
    llm.extract("uid1", "report")
    llm_forced = LLMExtractor(cache_dir=tmp_path, client=client, force=True)
    llm_forced.extract("uid1", "report")
    assert client.messages.calls == 2


def test_prompt_version_mismatch_invalidates(tmp_path):
    client = make_client(make_response())
    llm = LLMExtractor(cache_dir=tmp_path, client=client)
    llm.extract("uid1", "report")
    # Corrupt the cached prompt_version
    path = tmp_path / "uid1.json"
    entry = json.loads(path.read_text())
    entry["prompt_version"] = "stale"
    path.write_text(json.dumps(entry))
    llm.extract("uid1", "report")
    assert client.messages.calls == 2


def test_error_sentinel_no_crash(tmp_path):
    client = make_client(RuntimeError("boom"))
    llm = LLMExtractor(cache_dir=tmp_path, client=client)
    result = llm.extract("uid1", "report")
    assert result.error
    assert (tmp_path / "uid1.json").exists()


def test_refusal_sentinel(tmp_path):
    client = make_client(make_response(stop_reason="refusal"))
    llm = LLMExtractor(cache_dir=tmp_path, client=client)
    result = llm.extract("uid1", "report")
    assert result.error == "refusal"


def test_uncached_uids(tmp_path):
    client = make_client(make_response())
    llm = LLMExtractor(cache_dir=tmp_path, client=client)
    llm.extract("uid1", "report")
    assert llm.uncached_uids(["uid1", "uid2"]) == ["uid2"]


def test_cost_estimate_sane():
    full = estimate_cost(4407, 1100, batch=True)
    assert 1.0 < full < 15.0
    assert estimate_cost(4407, 1100, batch=False) == pytest.approx(full * 2)


def test_schema_covers_all_labels():
    required = OUTPUT_SCHEMA["properties"]["labels"]["required"]
    assert required == list(TARGET_LABELS)
    assert PROMPT_VERSION
