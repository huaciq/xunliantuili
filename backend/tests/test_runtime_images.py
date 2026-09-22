import uuid

from fastapi.testclient import TestClient

from app.main import app


def _login(client: TestClient, email: str, password: str) -> dict[str, str]:
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_admin_can_register_update_and_activate_runtime_image() -> None:
    with TestClient(app) as client:
        admin = _login(client, "admin@example.com", "AdminPass123!")
        suffix = uuid.uuid4().hex[:8]
        created = client.post(
            "/api/v1/admin/runtime-images",
            headers=admin,
            json={
                "name": f"INP Runtime {suffix}",
                "image": f"train-platform/inpformer:{suffix}",
                "digest": "development",
                "framework": "INPFormer",
                "is_active": False,
            },
        )
        assert created.status_code == 201
        runtime = created.json()
        assert runtime["framework"] == "inpformer"
        assert runtime["is_active"] is False
        public_before = client.get("/api/v1/runtime-images", headers=admin).json()
        assert runtime["id"] not in {item["id"] for item in public_before}

        updated = client.patch(
            f"/api/v1/admin/runtime-images/{runtime['id']}",
            headers=admin,
            json={"is_active": True},
        )
        assert updated.status_code == 200
        assert updated.json()["is_active"] is True
        public_after = client.get("/api/v1/runtime-images", headers=admin).json()
        assert runtime["id"] in {item["id"] for item in public_after}

        email = f"runtime-{suffix}@example.com"
        client.post(
            "/api/v1/users",
            headers=admin,
            json={"email": email, "name": "Runtime User", "password": "UserPass123!"},
        )
        user = _login(client, email, "UserPass123!")
        forbidden = client.get("/api/v1/admin/runtime-images", headers=user)
        assert forbidden.status_code == 403
