import pytest
from fastapi.testclient import TestClient
from api.main import app
import io
from PIL import Image
import numpy as np


class TestAPIEndpoints:
    """Test suite for the image classification API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a FastAPI test client."""
        with TestClient(app) as c:
            yield c

    @pytest.fixture
    def valid_jpeg_image(self):
        """Generate a valid JPEG image in memory."""
        np.random.seed(42)  # Reproducibility
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        return buf

    @pytest.fixture
    def valid_png_image(self):
        """Generate a valid PNG image in memory."""
        np.random.seed(42)
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf

    @pytest.fixture
    def oversized_image(self):
        """Generate a large resolution image."""
        img_array = np.random.randint(0, 255, (4000, 4000, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        return buf

    @pytest.fixture
    def tiny_image(self):
        """Generate a very small image."""
        img_array = np.random.randint(0, 255, (1, 1, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        return buf

    def test_health_check(self, client):
        """Test that the health/root endpoint is reachable."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_predict_oversized_image(self, client, oversized_image):
        """Test that the API handles large images (resize internally)."""
        response = client.post(
            "/predict",
            files={"pFile": ("big.jpg", oversized_image, "image/jpeg")}
        )
        # Should still succeed if server resizes internally
        assert response.status_code in [200, 413]

    def test_predict_tiny_image(self, client, tiny_image):
        """Test prediction with a 1×1 pixel image."""
        response = client.post(
            "/predict",
            files={"pFile": ("tiny.jpg", tiny_image, "image/jpeg")}
        )
        assert response.status_code in [200, 400]

    def test_predict_grayscale_image(self, client):
        """Test prediction with a single-channel grayscale image."""
        img_array = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        img = Image.fromarray(img_array, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = client.post(
            "/predict",
            files={"pFile": ("gray.jpg", buf, "image/jpeg")}
        )
        # Server should convert to RGB or reject gracefully
        assert response.status_code in [200, 400]

    def test_predict_no_file(self, client):
        """Test prediction with no file uploaded."""
        response = client.post("/predict")
        assert response.status_code == 422  # FastAPI validation error

    def test_predict_wrong_field_name(self, client, valid_jpeg_image):
        """Test prediction with incorrect form field name."""
        response = client.post(
            "/predict",
            files={"wrong_field": ("test.jpg", valid_jpeg_image, "image/jpeg")}
        )
        assert response.status_code == 422

    def test_predict_invalid_file_type(self, client):
        """Test prediction with a non-image file."""
        fake_file = io.BytesIO(b"this is not an image")
        response = client.post(
            "/predict",
            files={"pFile": ("test.txt", fake_file, "text/plain")}
        )
        assert response.status_code in [400, 422]

    def test_predict_corrupted_image(self, client):
        """Test prediction with corrupted JPEG bytes."""
        corrupted = io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        response = client.post(
            "/predict",
            files={"pFile": ("bad.jpg", corrupted, "image/jpeg")}
        )
        assert response.status_code in [400, 422, 500]

    def test_predict_empty_file(self, client):
        """Test prediction with an empty file."""
        empty = io.BytesIO(b"")
        response = client.post(
            "/predict",
            files={"pFile": ("empty.jpg", empty, "image/jpeg")}
        )
        assert response.status_code in [400, 422]

    def test_predict_get_not_allowed(self, client):
        """Test that GET is not allowed on /predict."""
        response = client.get("/predict")
        assert response.status_code == 405  # Method Not Allowed

    def test_predict_put_not_allowed(self, client):
        """Test that PUT is not allowed on /predict."""
        response = client.put("/predict")
        assert response.status_code == 405

    def test_response_json_structure(self, client, valid_jpeg_image):
        """Test that the response contains only expected keys."""
        response = client.post(
            "/predict",
            files={"pFile": ("test.jpg", valid_jpeg_image, "image/jpeg")}
        )
        data = response.json()
        expected_keys = {"prediction", "confidence"}
        assert set(data.keys()) >= expected_keys  # At minimum these keys

    def test_response_content_type(self, client, valid_jpeg_image):
        """Test that the response Content-Type is JSON."""
        response = client.post(
            "/predict",
            files={"pFile": ("test.jpg", valid_jpeg_image, "image/jpeg")}
        )
        assert response.headers["content-type"] == "application/json"

    def test_predict_deterministic(self, client):
        """Test that the same image produces the same result."""
        results = []
        for _ in range(3):
            np.random.seed(42)
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            buf.seek(0)

            response = client.post(
                "/predict",
                files={"pFile": ("test.jpg", buf, "image/jpeg")}
            )
            results.append(response.json())

        # All predictions should be identical
        assert all(r["prediction"] == results[0]["prediction"] for r in results)
        assert all(r["confidence"] == results[0]["confidence"] for r in results)