import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_classify_clear_ticket(client):
    resp = client.post("/classify", json={"text": "printer showing offline even though its on"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["issue_type"] == "printer_not_working"
    assert body["response_time_ms"] < 2000
    assert body["suggested_resolution"] is not None


def test_classify_noisy_ticket(client):
    resp = client.post("/classify", json={
        "text": "vpn keps disconecting evry few minuts pls help asap"
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["issue_type"] == "vpn_connection_failure"


def test_classify_rejects_empty_text(client):
    resp = client.post("/classify", json={"text": "a"})
    assert resp.status_code == 422


def test_response_time_budget(client):
    resp = client.post("/classify", json={"text": "outlook keeps crashing on startup"})
    assert resp.json()["response_time_ms"] < 2000
