import re
import cv2
import base64
import numpy as np
import pathlib
from typing import Dict, Any

def decode_image(image_data: Any):
    """Robustly decode image data (base64 str or bytes) into OpenCV format."""
    try:
        if isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("OpenCV could not decode image bytes")
            return img

        if not isinstance(image_data, str):
            raise ValueError(f"Unsupported image type: {type(image_data)}")

        clean_base64 = image_data.strip()

        if clean_base64.startswith("data:image"):
            match = re.search(r",(.+)$", clean_base64)
            if match:
                clean_base64 = match.group(1)
            else:
                comma_idx = clean_base64.find(",")
                if comma_idx != -1:
                    clean_base64 = clean_base64[comma_idx+1:]

        try:
            image_bytes = base64.b64decode(clean_base64)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 string: {str(e)}")

        nparr = np.frombuffer(image_bytes, np.uint8)
        if nparr.size == 0:
            raise ValueError("Image buffer is empty")

        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("OpenCV could not decode image")
        return img

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Decoding error: {str(e)}")


def clean_text(text: str, strip_html: bool = True) -> str:
    """Clean up internal PaddleOCR tags."""
    if not text:
        return ""

    text = text.replace("<fcel>", " | ")
    text = text.replace("<nl>", "\n")
    text = text.replace("<frow>", "\n")

    if strip_html:
        text = re.sub(r'<[^>]+>', '', text)

    return text.strip()


def generate_markdown(raw_result: Dict[str, Any]) -> str:
    """Extract or generate markdown content from OCR results."""
    markdown_content = raw_result.get('markdown', "")
    if markdown_content:
        return clean_text(markdown_content, strip_html=False)

    p_list = []
    if isinstance(raw_result, dict):
        p_list = raw_result.get('parsing_res_list')
        if p_list is None:
            p_list = []

        if not p_list:
            l_res = raw_result.get('layoutParsingResults')
            if l_res and len(l_res) > 0:
                res = l_res[0].get('prunedResult', {})
                p_list = res.get('parsing_res_list', [])

    if not p_list:
        return ""

    md_lines = []
    for item in p_list:
        item_data = (item if isinstance(item, dict)
                     else (vars(item) if hasattr(item, '__dict__') else {}))
        content = (item_data.get('content') or 
                   item_data.get('block_content', ""))
        if content:
            content = clean_text(content)
            block_type = item_data.get('type', 'text').lower()
            if block_type == 'header':
                md_lines.append(f"# {content}")
            elif block_type == 'title':
                md_lines.append(f"## {content}")
            elif block_type == 'table':
                md_lines.append(clean_text(content, strip_html=False))
            else:
                md_lines.append(clean_text(content, strip_html=True))
            md_lines.append("") 
    return "\n".join(md_lines)


def setup_model_cache():
    """Setup model cache directories in the volume."""
    # Create all necessary cache directories in the volume
    cache_dirs = [
        "/models/paddlex/official_models",
        "/models/paddlex/models",
        "/models/paddle",
        "/models/paddleocr",
        "/models/cache",
        "/models/huggingface",
        "/models/huggingface/transformers",
    ]

    for dir_path in cache_dirs:
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created cache directory: {dir_path}")
