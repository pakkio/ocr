from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class OCRConfig(BaseSettings):
    """Configuration for OCR application using dependency injection pattern"""
    
    # OpenRouter Configuration
    openrouter_api_key: Optional[str] = Field(default=None, env="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", env="OPENROUTER_BASE_URL")
    
    # Direct API Keys (fallback)
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    
    # OCR Settings
    default_ocr_prompt: str = Field(
        default="Extract all text from this image. Return only the text content, preserving formatting and structure.",
        env="DEFAULT_OCR_PROMPT"
    )
    structured_ocr_prompt: str = Field(
        default="Analyze this dashboard/analytics image and extract ALL visible data into structured JSON format. Focus on: numeric values, chart data points, time series data, key metrics, chart titles. Ignore watermarks and stock photo identifiers.",
        env="STRUCTURED_OCR_PROMPT"
    )
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    timeout_seconds: int = Field(default=30, env="TIMEOUT_SECONDS")
    
    # Available VLM Models via OpenRouter
    available_models: dict = Field(default={
        "gpt-4-vision-preview": {
            "name": "GPT-4 Vision Preview",
            "provider": "openai",
            "cost_per_1k_tokens": 0.01,
            "supports_vision": True
        },
        "gpt-4o": {
            "name": "GPT-4o",
            "provider": "openai", 
            "cost_per_1k_tokens": 0.005,
            "supports_vision": True
        },
        "claude-3-5-sonnet-20241022": {
            "name": "Claude 3.5 Sonnet",
            "provider": "anthropic",
            "cost_per_1k_tokens": 0.003,
            "supports_vision": True
        },
        "claude-3-5-haiku-20241022": {
            "name": "Claude 3.5 Haiku",
            "provider": "anthropic",
            "cost_per_1k_tokens": 0.00025,
            "supports_vision": True
        },
        "google/gemini-pro-1.5": {
            "name": "Gemini Pro 1.5",
            "provider": "google",
            "cost_per_1k_tokens": 0.00125,
            "supports_vision": True
        },
        "google/gemini-flash-1.5": {
            "name": "Gemini Flash 1.5",
            "provider": "google",
            "cost_per_1k_tokens": 0.000075,
            "supports_vision": True
        },
        "mistralai/pixtral-12b": {
            "name": "Mistral Pixtral 12B",
            "provider": "mistral",
            "cost_per_1k_tokens": 0.0015,
            "supports_vision": True
        },
        "qwen/qwen-2-vl-72b-instruct": {
            "name": "Qwen2-VL 72B",
            "provider": "qwen",
            "cost_per_1k_tokens": 0.0009,
            "supports_vision": True
        },
        "qwen/qwen-2-vl-7b-instruct": {
            "name": "Qwen2-VL 7B",
            "provider": "qwen",
            "cost_per_1k_tokens": 0.0002,
            "supports_vision": True
        }
    })
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Singleton config instance
config = OCRConfig()