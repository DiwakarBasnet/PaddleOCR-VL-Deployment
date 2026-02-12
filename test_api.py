import asyncio
import base64
import time
import argparse
import aiohttp
import json
import os
import glob
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont


def create_test_image(text="Hello Modal OCR"):
    """Creates a simple image with text for testing."""
    width = 400
    height = 200
    color = (255, 255, 255)
    img = Image.new('RGB', (width, height), color)
    d = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    d.text((50, 80), text, fill=(0, 0, 0), font=font)

    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def load_images_from_dir(directory):
    """Loads all images from a directory and converts them to base64."""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []

    if not os.path.exists(directory):
        print(f"Warning: Directory '{directory}' not found.")
        return []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
        # Also try case-insensitive
        image_files.extend(glob.glob(os.path.join(directory, ext.upper())))

    # Remove duplicates
    image_files = sorted(list(set(image_files)))

    images_b64 = []
    print(f"Found {len(image_files)} images in '{directory}'")

    for img_path in image_files:
        try:
            with open(img_path, "rb") as image_file:
                b64_string = base64.b64encode(
                    image_file.read()).decode('utf-8')
                images_b64.append(b64_string)
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")

    return images_b64


async def send_request(session, url, image_b64, req_id):
    """Sends a single request to the API."""
    payload = {
        "image_base64": image_b64,
        "prompt": "ocr",
        "use_layout_detection": True,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": True,
        "layout_merge_bboxes_mode": "small"
    }

    print(f"Request {req_id}: Sending...")
    start_time = time.time()

    try:
        async with session.post(url, json=payload, timeout=600) as response:
            latency = (time.time() - start_time) * 1000
            if response.status == 200:
                data = await response.json()
                print(f"Request {req_id}: Success in {latency:.2f}ms")
                return {
                    "id": req_id,
                    "success": True,
                    "status": response.status,
                    "latency": latency,
                    "data": data,
                    "error": None
                }
            else:
                text = await response.text()
                print(
                    f"Request {req_id}: Failed ({response.status}) in {latency:.2f}ms")
                return {
                    "id": req_id,
                    "success": False,
                    "status": response.status,
                    "latency": latency,
                    "data": None,
                    "error": text
                }
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        print(f"Request {req_id}: Exception in {latency:.2f}ms: {str(e)}")
        return {
            "id": req_id,
            "success": False,
            "status": 0,
            "latency": latency,
            "data": None,
            "error": str(e)
        }


async def run_load_test(url, num_requests, concurrency, image_dir=None):
    """Runs the concurrent load test."""
    images = []
    if image_dir:
        images = load_images_from_dir(image_dir)

    if not images:
        if image_dir:
            print("No images found in directory, falling back to synthetic image.")
        else:
            print("No image directory specified, using synthetic image.")
        images = [create_test_image("Concurrent OCR Test")]

    print(f"Starting load test against {url}")
    print(f"Total Requests: {num_requests}, Concurrency: {concurrency}")
    print(f"Using {len(images)} unique images (cycling through them)")

    results = []

    async with aiohttp.ClientSession() as session:
        # Create batches of tasks based on concurrency
        for i in range(0, num_requests, concurrency):
            batch_size = min(concurrency, num_requests - i)
            tasks = []
            for j in range(batch_size):
                req_id = i + j + 1
                # Cycle through available images
                img_b64 = images[(req_id - 1) % len(images)]
                tasks.append(send_request(session, url, img_b64, req_id))

            # Run batch
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

    return results


def save_results_to_markdown(results, output_file):
    """Saves test results to a markdown file."""
    success_count = sum(1 for r in results if r['success'])
    total_time = sum(r['latency'] for r in results)
    avg_latency = total_time / len(results) if results else 0
    max_latency = max((r['latency'] for r in results), default=0)
    min_latency = min((r['latency'] for r in results), default=0)

    with open(output_file, 'w') as f:
        f.write(f"# API Load Test Results\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Requests:** {len(results)}\n")
        f.write(
            f"**Success Rate:** {success_count}/{len(results)} ({products_pct(success_count, len(results))})\n")
        f.write(f"**Average Latency:** {avg_latency:.2f} ms\n")
        f.write(
            f"**Min/Max Latency:** {min_latency:.2f} ms / {max_latency:.2f} ms\n\n")

        f.write("## Detailed Results\n\n")
        f.write("| ID | Status | Latency (ms) | Success | Error |\n")
        f.write("|----|--------|--------------|---------|-------|\n")

        for r in results:
            error_msg = r['error'] if r['error'] else ""
            # Truncate error if too long
            if error_msg and len(error_msg) > 50:
                error_msg = error_msg[:47] + "..."
            f.write(
                f"| {r['id']} | {r['status']} | {r['latency']:.2f} | {'✅' if r['success'] else '❌'} | {error_msg} |\n")

        f.write("\n## Markdown Outputs (Successes)\n\n")
        for r in results:
            if r['success'] and r['data']:
                f.write(f"### Request {r['id']}\n")
                if 'markdown' in r['data']:
                    f.write("```markdown\n")
                    f.write(r['data']['markdown'])
                    f.write("\n```\n")
                else:
                    f.write("_No markdown field in response._\n")


def products_pct(part, whole):
    return f"{100 * float(part)/float(whole):.1f}%" if whole else "0%"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Modal OCR API")
    parser.add_argument("--url", type=str, required=True,
                        help="API Endpoint URL")
    parser.add_argument("--requests", type=int, default=5,
                        help="Total number of requests")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Number of concurrent requests")
    parser.add_argument("--image-dir", type=str, default="test_images",
                        help="Directory containing test images")
    parser.add_argument(
        "--output", type=str, default="api_test_results.md", help="Output Markdown file")

    args = parser.parse_args()

    try:
        results = asyncio.run(run_load_test(
            args.url, args.requests, args.concurrency, args.image_dir))
        save_results_to_markdown(results, args.output)
        print(f"\nResults saved to {args.output}")
    except KeyboardInterrupt:
        print("\nTest cancelled.")
