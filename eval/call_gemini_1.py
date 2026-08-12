import requests
import base64
import json
import time
import os
from pathlib import Path

API_URL = "http://115.120.87.159:8082/v1beta/models/gemini-3.5-flash:generateContent"
API_KEY_FILE = "./api_key.txt"

# video_file_name = "./videos/demo_007_a.mp4"
# video_file_name = "./videos/demo_007.mp4"
# video_file_name = "./videos/demo_008.mp4"
video_file_name = "./videos/demo_006.mp4"
question = "不要借助现写的代码工具，你直接看这个视频，告诉我视频里的球体是不是静止的？如果不是，具体是怎么样运动的，往哪个方向？"

# video_file_name = "./videos/translation_006.mp4"
# question = "不要借助现写的代码工具，你直接看这个视频，告诉我视频里的画面是不是静止的？如果不是，具体是怎么样运动的，往哪个方向？"

SAVE_PATH = "./results/gemini-3.5-flash.jsonl"


def read_api_key(key_file: str) -> str:
    """从指定 txt 文件读取 API key，文件中仅需放置 key 本身。"""
    path = Path(key_file).expanduser()
    try:
        api_key = path.read_text(encoding="utf-8-sig").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(f"API key 文件不存在：{path}") from exc
    except OSError as exc:
        raise RuntimeError(f"无法读取 API key 文件：{path}（{exc}）") from exc

    if not api_key:
        raise RuntimeError(f"API key 文件为空：{path}")
    return api_key


def generate():
    api_key = read_api_key(API_KEY_FILE)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    with open(video_file_name, "rb") as f:
        video_bytes = f.read()

    payload = {
        "contents": [{
            "parts": [
                {"text": question},
                {
                    "inline_data": {
                        "mime_type": "video/mp4",
                        "data": base64.b64encode(video_bytes).decode("utf-8")
                    }
                }
            ]
        }]
    }

    response = requests.post(f"{API_URL}", json=payload, headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        text = response_data["candidates"][0]["content"]["parts"][0]["text"]
        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "video_path": video_file_name,
            "question": question,
            "model_answer": text,
            "finish_reason": response_data["candidates"][0].get("finishReason"),
        }

        print(json.dumps(result, indent=2, ensure_ascii=False))

        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        with open(SAVE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"\n✅ 已保存到 {SAVE_PATH}")
    else:
        print(f"请求失败: {response.status_code}, {response.text}")


if __name__ == "__main__":
    generate()