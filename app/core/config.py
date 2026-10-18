import os
import logging
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "LazarTrack Intelligent Extraction Agent"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"

    EASYOCR_URL: str = os.getenv("EASYOCR_URL", "").strip()
    MAX_IMAGE_BYTES: int = 20 * 1024 * 1024  # 20 MB
    
    # Video pipeline configuration
    VIDEO_OUTPUT_DIR: str = os.getenv(
        "VIDEO_OUTPUT_DIR", 
        os.path.join(os.path.dirname(__file__), "..", "..", "video", "output")
    )
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

    PLACEHOLDER_URL: str = "http://your-easyocr-server:PORT"

    def validate(self):
        if not self.EASYOCR_URL or self.EASYOCR_URL == self.PLACEHOLDER_URL:
            raise RuntimeError(
                "EASYOCR_URL environment variable is not configured correctly in .env!"
            )
        if not self.OPENROUTER_API_KEY:
            logging.warning("OPENROUTER_API_KEY not set - battle log extraction (step 7) will fail")

settings = Settings()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    )
