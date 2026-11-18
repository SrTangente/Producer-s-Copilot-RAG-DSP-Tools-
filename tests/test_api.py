import io
import json

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import create_app
from app.references import ReferenceStore
from tests.conftest import SR, drum_loop

WS = {"X-Workspace": "test-workspace-123"}


def wav_bytes(y, sr=SR):
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


@pytest.fixture(scope="module")
def loop_wav():
    return wav_bytes(drum_loop(seconds=6))


@pytest.fixture
def client(tmp_path):
    app = create_app(store=ReferenceStore(tmp_path / "refs.json"), writer=None)
    return TestClient(app)


def upload(data, name="mix.wav"):
    return {"file": (name, data, "audio/wav")}


def test_health_and_index(client):
    assert client.get("/health").json()["tip_engine"] == "rules"
    page = client.get("/")
    assert page.status_code == 200 and "Producer" in page.text


def test_analyze(client, loop_wav):
    r = client.post("/analyze", files=upload(loop_wav))
    assert r.status_code == 200
    assert r.json()["bpm"] == pytest.approx(120, abs=3)


def test_rejects_bad_uploads(client):
    assert client.post("/analyze", files=upload(b"abc", "notes.txt")).status_code == 415
    assert client.post("/analyze", files=upload(b"garbage", "fake.wav")).status_code == 422


def test_reference_lifecycle(client, loop_wav):
    assert client.post("/references", files=upload(loop_wav)).status_code == 400  # no workspace

    ref = client.post("/references", files=upload(loop_wav, "Daft Ref.wav"), headers=WS).json()
    assert ref["name"] == "Daft Ref"
    assert "lufs_i" in ref["metrics"]

    listed = client.get("/references", headers=WS).json()["references"]
    assert [r["id"] for r in listed] == [ref["id"]]
    # Other workspaces can't see it
    assert client.get("/references", headers={"X-Workspace": "someone-else-999"}).json()["references"] == []

    assert client.delete(f"/references/{ref['id']}", headers=WS).status_code == 200
    assert client.get("/references", headers=WS).json()["references"] == []


def test_tips_with_uploaded_reference(client, loop_wav):
    ref = client.post("/references", files=upload(loop_wav, "ref.wav"), headers=WS).json()
    quiet = wav_bytes(drum_loop(seconds=6, gain=0.08))
    r = client.post("/tips", files=upload(quiet), data={"reference_id": ref["id"]}, headers=WS)
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "rules"
    assert body["reference_name"] == "ref"
    assert any(f["id"] == "quieter_than_ref" for f in body["findings"])
    assert body["comparison"] and body["tips"] and body["sources"]


def test_tips_with_reference_held_by_the_browser(client, loop_wav):
    metrics = client.post("/analyze", files=upload(loop_wav)).json()
    quiet = wav_bytes(drum_loop(seconds=6, gain=0.08))
    r = client.post("/tips", files=upload(quiet),
                    data={"reference_json": json.dumps(metrics), "reference_name": 'My <ref>"'})
    assert r.status_code == 200
    body = r.json()
    assert body["reference_name"] == "My ref"
    assert any(f["id"] == "quieter_than_ref" for f in body["findings"])
    assert any(row["metric"] == "band_level_db.sub" for row in body["comparison"])


@pytest.mark.parametrize("bad", ["not json", "[1, 2]", "{}", '{"lufs_i": "loud"}', '{"lufs_i": 1e999}'])
def test_tips_rejects_bad_reference_data(client, loop_wav, bad):
    r = client.post("/tips", files=upload(loop_wav), data={"reference_json": bad})
    assert r.status_code == 422


def test_tips_with_genre_target(client, loop_wav):
    targets = client.get("/targets").json()["targets"]
    assert {"pop", "edm", "streaming"} <= {t["id"] for t in targets}
    r = client.post("/tips", files=upload(loop_wav), data={"target_id": "edm"})
    assert r.status_code == 200
    assert r.json()["reference_name"] == "the EDM / club profile"


def test_tips_unknown_reference(client, loop_wav):
    r = client.post("/tips", files=upload(loop_wav), data={"reference_id": "nope"}, headers=WS)
    assert r.status_code == 404
