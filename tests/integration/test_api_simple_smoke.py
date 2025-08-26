import os
import pytest
import requests

pytestmark = pytest.mark.skipif(os.getenv("E2E_LOCAL") != "1", reason="Local smoke test; set E2E_LOCAL=1 to run")


def test_api_simple_smoke():
    # health
    resp = requests.get('http://localhost:8000/health')
    assert resp.ok
    # health/db
    resp = requests.get('http://localhost:8000/health/db')
    assert resp.ok
    # upload
    with open('data/sample.csv', 'rb') as f:
        files = {'file': ('sample.csv', f, 'text/csv')}
        resp = requests.post('http://localhost:8000/api/v1/upload', files=files)
    assert resp.status_code in (200, 202)
