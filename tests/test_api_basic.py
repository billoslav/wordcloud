import pytest # type: ignore

from wordcloud.api import create_app, FLASK_AVAILABLE


@pytest.mark.skipif(not FLASK_AVAILABLE, reason="Flask not installed")
def test_api_version():
    app = create_app()
    client = app.test_client()
    resp = client.get("/api/version")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["name"] == "wordcloud-api"


@pytest.mark.skipif(not FLASK_AVAILABLE, reason="Flask not installed")
def test_api_generate_and_download(tmp_path):
    app = create_app()
    client = app.test_client()

    resp = client.post("/api/generate", json={"text": "hello", "width": 50, "height": 50})
    assert resp.status_code == 202  # Accepted (async job)
    job_id = resp.get_json()["job_id"]

    # poll status
    import time

    for _ in range(200):
        status_resp = client.get(f"/api/status/{job_id}")
        status = status_resp.get_json()["status"]
        if status == "completed":
            break
        if status == "failed":
            pytest.fail("Job failed")
        time.sleep(0.05)
    else:
        pytest.fail("Job did not complete")

    dl = client.get(f"/api/download/{job_id}")
    assert dl.status_code == 200
    assert dl.mimetype == "image/png"

