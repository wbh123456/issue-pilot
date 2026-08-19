"""Hidden gold test for issue-017 — evaluator only."""

from fastapi.testclient import TestClient

from app import catalog, shipping
from app.main import app

client = TestClient(app, raise_server_exceptions=False)


def test_download_only_title_is_not_billed_for_delivery():
    items = [{"sku": "ebook", "qty": 1}]
    assert catalog.digital_only(items) is True
    assert shipping.quote_shipment(items) == 0.0
    quoted = client.get("/quotes/delivery", params={"sku": "ebook"})
    assert quoted.status_code == 200
    assert quoted.json()["postage"] == 0.0
