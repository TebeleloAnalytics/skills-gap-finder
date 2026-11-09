import pytest
from fastapi.testclient import TestClient

from skills_gap_finder.api import app, get_service


class FakeService:
    def roles(self):
        return ["data scientist"]

    def search(self, query, k):
        return [{"label": "statistics", "similarity": 0.7}][:k]

    def gap_analysis(self, statements, role):
        if role != "data scientist":
            raise KeyError(f"Unknown role {role!r}")
        return {"role": role, "essential_coverage": 0.5, "covered": [], "gaps": []}

    def describe(self, uris):
        return []


@pytest.fixture
def client():
    app.dependency_overrides[get_service] = FakeService
    # no `with` block: the startup hook (which loads the real model) doesn't run
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_roles_and_search(client):
    assert client.get("/roles").json() == ["data scientist"]
    assert client.get("/search", params={"q": "I analyse data", "k": 1}).json()[0]["label"] == "statistics"


def test_search_validates_input(client):
    assert client.get("/search", params={"q": "x"}).status_code == 422  # too short


def test_gap_unknown_role_is_404(client):
    response = client.post("/gap", json={"role": "astronaut", "statements": ["I fly"]})
    assert response.status_code == 404


def test_gap_requires_statements(client):
    assert client.post("/gap", json={"role": "data scientist", "statements": []}).status_code == 422


def test_plan_without_gaps(client):
    response = client.post("/plan", json={"role": "data scientist", "statements": ["I analyse data"]})
    assert response.status_code == 200
    assert response.json()["plan"] == "No essential gaps found for this role."
