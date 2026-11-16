from fastapi.testclient import TestClient

from app.main import app


def auth_header(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@example.com", "password": "AdminPass123!"},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_admin_can_create_user_and_project() -> None:
    with TestClient(app) as client:
        headers = auth_header(client)
        user_response = client.post(
            "/api/v1/users",
            headers=headers,
            json={
                "email": "researcher@example.com",
                "name": "Researcher",
                "password": "Research123!",
                "system_role": "user",
            },
        )
        assert user_response.status_code in (201, 409)

        project_response = client.post(
            "/api/v1/projects",
            headers=headers,
            json={"name": "Defect Detection", "description": "Visual anomaly experiments"},
        )
        assert project_response.status_code == 201
        project = project_response.json()
        assert project["name"] == "Defect Detection"
        assert project["current_user_role"] == "project_owner"

        member_response = client.post(
            f"/api/v1/projects/{project['id']}/members",
            headers=headers,
            json={"email": "researcher@example.com", "role": "researcher"},
        )
        assert member_response.status_code == 200
        assert member_response.json()["role"] == "researcher"
