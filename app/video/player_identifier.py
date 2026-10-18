"""
Step 3: Player Identification Module using Vision Language Model via Open Router.

Extracts player names from team_slots first frame using a Vision Language model.
"""

import json
import logging
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import base64
import httpx

log = logging.getLogger("app.video.player_identifier")

# ========== CONFIGURATION ==========
VL_MODEL = "qwen/qwen3.7-flash"  # Changed from qwen-3.7-flash (doesn't exist)
VL_API_URL = "https://openrouter.ai/api/v1/chat/completions"
VL_TEMPERATURE = 0.1
VL_MAX_TOKENS = 500

VL_PROMPT = (
    "This is a Free Fire battle royale game screenshot showing the team slots UI. "
    "Read the exact in-game player names visible on the screen. "
    "Return only the names you can actually see — use null for any slot that is empty or not visible. "
    "Preserve exact capitalisation and special characters."
)

# Structured JSON Schema — slots are optional so the model won't hallucinate missing ones
VL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "1": {"type": ["string", "null"], "description": "Exact player name in slot 1, or null if not visible"},
        "2": {"type": ["string", "null"], "description": "Exact player name in slot 2, or null if not visible"},
        "3": {"type": ["string", "null"], "description": "Exact player name in slot 3, or null if not visible"},
        "4": {"type": ["string", "null"], "description": "Exact player name in slot 4, or null if not visible"},
    },
    "required": ["1", "2", "3", "4"],
    "additionalProperties": False,
}
# ========== END CONFIGURATION =========


class VisionLanguageModel:
    """Vision Language Model client using Open Router API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize VL model client."""
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set in .env")
        self.model = VL_MODEL
        self.client = httpx.Client(timeout=60.0)

    def extract_players_from_image(self, image: np.ndarray) -> Tuple[Dict[int, str], float]:
        """
        Extract player names from image.
        
        :param image: BGR image (team_slots frame)
        :return: Tuple of (Dict[slot_num -> player_name], confidence)
        """
        # Convert image to base64 data URI
        _, buffer = cv2.imencode(".png", image)
        image_b64 = base64.b64encode(buffer).decode("utf-8")
        data_uri = f"data:image/png;base64,{image_b64}"

        try:
            response = self.client.post(
                VL_API_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": VL_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": VL_PROMPT},
                                {"type": "image_url", "image_url": {"url": data_uri}},
                            ],
                        }
                    ],
                    "temperature": VL_TEMPERATURE,
                    "max_tokens": VL_MAX_TOKENS,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "player_extraction",
                            "schema": VL_RESPONSE_SCHEMA,
                            "strict": True,
                        },
                    },
                },
            )

            if response.status_code != 200:
                log.warning(f"API error {response.status_code}: {response.text}")
                return {}, 0.0

            content = response.json()["choices"][0]["message"]["content"].strip()
            log.info(f"Raw model response: {content}")

            # Parse JSON (guaranteed valid via structured output)
            parsed = json.loads(content)

            if not parsed:
                return {}, 0.0

            # Convert to int keys, remove nulls
            players = {}
            for key, value in parsed.items():
                if value is not None:
                    players[int(key)] = value

            log.info(f"Extracted {len(players)} players")
            return players, 0.85  # Default confidence

        except Exception as e:
            log.error(f"VL model error: {e}")
            return {}, 0.0

    def close(self):
        """Close HTTP client."""
        self.client.close()


class PlayerIdentifier:
    """Extracts player names from team_slots frames."""

    def __init__(self, vl_model: Optional[VisionLanguageModel] = None, output_dir: str = "Backend/video/output/step3"):
        """Initialize PlayerIdentifier."""
        if vl_model is None:
            vl_model = VisionLanguageModel()
        self.vl_model = vl_model
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def process_video_step2_output(self, step2_video_dir: str) -> Dict:
        """
        Extract players from Step 2 output and save JSON.
        
        :param step2_video_dir: Path to step2/{video_id}
        :return: Dict with video_id and players
        """
        video_id = os.path.basename(step2_video_dir.rstrip("/"))
        frame_path = os.path.join(step2_video_dir, "team_slots", "frames", "frame_0000.png")

        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Frame not found: {frame_path}")

        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Failed to read frame: {frame_path}")

        # Extract players
        players, confidence = self.vl_model.extract_players_from_image(frame)

        # Save JSON
        video_dir = os.path.join(self.output_dir, video_id)
        os.makedirs(video_dir, exist_ok=True)
        
        output_json = os.path.join(video_dir, "players.json")
        with open(output_json, "w") as f:
            json.dump(players, f, indent=2)

        log.info(f"[Video {video_id}] Saved {len(players)} players")
        return {"video_id": video_id, "players": players, "confidence": confidence}
