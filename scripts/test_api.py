"""
Script test API dự đoán /predict của dịch vụ WDBC Breast Cancer FastAPI
"""
import urllib.request
import json
import sys

API_URL = "http://localhost:18000/predict"
HEALTH_URL = "http://localhost:18000/health"

sample_data = {
    "features": {
        "mean_radius": 17.99,
        "mean_texture": 10.38,
        "mean_perimeter": 122.8,
        "mean_area": 1001.0,
        "mean_smoothness": 0.1184,
        "mean_compactness": 0.2776,
        "mean_concavity": 0.3001,
        "mean_concave_points": 0.1471,
        "mean_symmetry": 0.2419,
        "mean_fractal_dimension": 0.07871,
        "radius_error": 1.095,
        "texture_error": 0.9053,
        "perimeter_error": 8.589,
        "area_error": 153.4,
        "smoothness_error": 0.006399,
        "compactness_error": 0.04904,
        "concavity_error": 0.05373,
        "concave_points_error": 0.01587,
        "symmetry_error": 0.03003,
        "fractal_dimension_error": 0.006193,
        "worst_radius": 25.38,
        "worst_texture": 17.33,
        "worst_perimeter": 184.6,
        "worst_area": 2019.0,
        "worst_smoothness": 0.1622,
        "worst_compactness": 0.6656,
        "worst_concavity": 0.7119,
        "worst_concave_points": 0.2654,
        "worst_symmetry": 0.4601,
        "worst_fractal_dimension": 0.1189
    }
}

def main():
    print(f"1. Checking API Health ({HEALTH_URL})...")
    try:
        req = urllib.request.Request(HEALTH_URL)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"Health status: {data}")
    except Exception as e:
        print(f"Health check failed: {e}")
        sys.exit(1)

    print(f"\n2. Sending sample prediction request to ({API_URL})...")
    try:
        req = urllib.request.Request(
            API_URL,
            data=json.dumps(sample_data).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode())
            print("Prediction Response:")
            print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Prediction request failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
