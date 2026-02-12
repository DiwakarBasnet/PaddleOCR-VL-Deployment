import argparse
import base64
import json
import urllib.request
import urllib.error
import sys
import os
import time

# The API endpoint provided by the user
API_URL = "https://diwakarbasnet--paddleocr-vl-fastapi-app.modal.run/predict"


def main():
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR-VL inference via HTTP API.")
    parser.add_argument("image_path", help="Path to the image file")
    args = parser.parse_args()

    # check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at {args.image_path}")
        sys.exit(1)

    print(f"Reading image: {args.image_path}")
    try:
        with open(args.image_path, "rb") as image_file:
            bnary_data = image_file.read()
            # API expects base64 string
            image_b64 = base64.b64encode(bnary_data).decode('utf-8')
    except Exception as e:
        print(f"Error reading image file: {e}")
        sys.exit(1)

    payload = {
        "image_base64": image_b64,
        "prompt": "ocr",
        "use_layout_detection": True,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": True,
        "layout_merge_bboxes_mode": "small"
    }

    print(f"Sending request to {API_URL}...")

    try:
        json_data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            API_URL,
            data=json_data,
            headers={'Content-Type': 'application/json'}
        )

        start_time = time.time()
        with urllib.request.urlopen(req) as response:
            resp_body = response.read()
            latency = (time.time() - start_time) * 1000

            result = json.loads(resp_body)

            if result.get("success"):
                print(f"Success! (Latency: {latency:.2f}ms)")
                print("\n" + "="*40)
                print("OCR Result (Markdown):")
                print("="*40 + "\n")
                print(result.get("markdown", ""))
                print("\n" + "="*40)
            else:
                print(f"OCR Operation reported failure: {result.get('error')}")

    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}")
        try:
            err_body = e.read().decode('utf-8')
            print(f"Response: {err_body}")
        except:
            pass
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Connection Error: {e.reason}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
