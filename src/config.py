from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv


print(f"Attempting to load .env from current working directory: {os.getcwd()}") 
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
            "supports_vision": True,
            "supports_strict_json_schema": True
        },
        "gpt-4o": {
            "name": "GPT-4o",
            "provider": "openai", 
            "cost_per_1k_tokens": 0.005,
            "supports_vision": True,
            "supports_strict_json_schema": False
        },
        "openai/gpt-4o-mini": {
            "name": "GPT-4o Mini",
            "provider": "openai",
            "cost_per_1k_tokens": 0.00015,
            "supports_vision": True,
            "supports_strict_json_schema": False
        },
        "openai/gpt-4.1": {
            "name": "GPT-4.1",
            "provider": "openai",
            "cost_per_1k_tokens": 0.008,
            "supports_vision": True,
            "supports_strict_json_schema": False
        },
        "openai/gpt-4.1-mini": {
            "name": "GPT-4.1 Mini",
            "provider": "openai",
            "cost_per_1k_tokens": 0.0002,
            "supports_vision": True,
            "supports_strict_json_schema": False
        },
        "openai/gpt-4.1-nano": {
            "name": "GPT-4.1 Nano",
            "provider": "openai",
            "cost_per_1k_tokens": 0.0001,
            "supports_vision": True,
            "supports_strict_json_schema": False
        },
        "anthropic/claude-3.5-sonnet": {
            "name": "Claude 3.5 Sonnet",
            "provider": "anthropic",
            "cost_per_1k_tokens": 0.003,
            "supports_vision": True,
            "supports_strict_json_schema": False
        },
        "anthropic/claude-sonnet-4": {
            "name": "Claude Sonnet 4",
            "provider": "anthropic",
            "cost_per_1k_tokens": 0.003,
            "supports_vision": True,
            "supports_strict_json_schema": False
        },
        "anthropic/claude-3.7-sonnet": {
            "name": "Claude 3.7 Sonnet",
            "provider": "anthropic",
            "cost_per_1k_tokens": 0.003,
            "supports_vision": True,
            "supports_strict_json_schema": False
        },
        "anthropic/claude-3.5-haiku": {
            "name": "Claude 3.5 Haiku",
            "provider": "anthropic",
            "cost_per_1k_tokens": 0.00025,
            "supports_vision": True,
            "supports_strict_json_schema": False
        },
        "google/gemini-2.0-flash-exp": {
            "name": "Gemini 2.0 Flash",
            "provider": "google",
            "cost_per_1k_tokens": 0.000075,
            "supports_vision": True,
            "supports_strict_json_schema": False
        },
        "google/gemini-exp-1114": {
            "name": "Gemini Experimental 1114",
            "provider": "google",
            "cost_per_1k_tokens": 0.000075,
            "supports_vision": True,
            "supports_strict_json_schema": False
        },
        "google/gemini-2.5-pro": {
            "name": "Gemini 2.5 Pro",
            "provider": "google",
            "cost_per_1k_tokens": 0.001,
            "supports_vision": True,
            "supports_strict_json_schema": False
        },
        "google/gemini-2.5-flash": {
            "name": "Gemini 2.5 Flash",
            "provider": "google",
            "cost_per_1k_tokens": 0.000075,
            "supports_vision": True,
            "supports_strict_json_schema": False
        },
        "google/gemini-2.5-flash-lite-preview-06-17": {
            "name": "Gemini 2.5 Flash Lite",
            "provider": "google",
            "cost_per_1k_tokens": 0.00005,
            "supports_vision": True,
            "supports_strict_json_schema": False
        },
        "mistralai/pixtral-12b": {
            "name": "Mistral Pixtral 12B",
            "provider": "mistral",
            "cost_per_1k_tokens": 0.0015,
            "supports_vision": True,
            "supports_strict_json_schema": False
        },
        "qwen/qwen-2-vl-72b-instruct": {
            "name": "Qwen2-VL 72B",
            "provider": "qwen",
            "cost_per_1k_tokens": 0.0009,
            "supports_vision": True,
            "supports_strict_json_schema": False
        },
        "qwen/qwen-2-vl-7b-instruct": {
            "name": "Qwen2-VL 7B",
            "provider": "qwen",
            "cost_per_1k_tokens": 0.0002,
            "supports_vision": True,
            "supports_strict_json_schema": False
        },
        "meta-llama/llama-3.2-11b-vision-instruct": {
            "name": "Llama 3.2 11B Vision",
            "provider": "meta",
            "cost_per_1k_tokens": 0.00055,
            "supports_vision": True,
            "supports_strict_json_schema": False
        },
        "meta-llama/llama-3.2-90b-vision-instruct": {
            "name": "Llama 3.2 90B Vision",
            "provider": "meta", 
            "cost_per_1k_tokens": 0.0018,
            "supports_vision": True,
            "supports_strict_json_schema": False
        },
        "microsoft/phi-4-multimodal-instruct": {
            "name": "Phi-4 Multimodal",
            "provider": "microsoft",
            "cost_per_1k_tokens": 0.0004,
            "supports_vision": True,
            "supports_strict_json_schema": False
        },
        "mistralai/pixtral-large-2411": {
            "name": "Pixtral Large",
            "provider": "mistral",
            "cost_per_1k_tokens": 0.002,
            "supports_vision": True,
            "supports_strict_json_schema": False
        }
    })
    
    # Model families for organized UI selection
    model_families: dict = Field(default={
        "OpenAI": {
            "gpt-4o": "GPT-4o",
            "openai/gpt-4o-mini": "GPT-4o Mini", 
            "openai/gpt-4.1": "GPT-4.1",
            "openai/gpt-4.1-mini": "GPT-4.1 Mini",
            "openai/gpt-4.1-nano": "GPT-4.1 Nano"
        },
        "Anthropic": {
            "anthropic/claude-3.5-sonnet": "Claude 3.5 Sonnet",
            "anthropic/claude-sonnet-4": "Claude Sonnet 4", 
            "anthropic/claude-3.7-sonnet": "Claude 3.7 Sonnet",
            "anthropic/claude-3.5-haiku": "Claude 3.5 Haiku"
        },
        "Google": {
            "google/gemini-2.5-pro": "Gemini 2.5 Pro",
            "google/gemini-2.5-flash": "Gemini 2.5 Flash",
            "google/gemini-2.5-flash-lite-preview-06-17": "Gemini 2.5 Flash Lite",
            "google/gemini-2.0-flash-exp": "Gemini 2.0 Flash",
            "google/gemini-exp-1114": "Gemini Experimental 1114"
        },
        "Meta": {
            "meta-llama/llama-3.2-11b-vision-instruct": "Llama 3.2 11B Vision",
            "meta-llama/llama-3.2-90b-vision-instruct": "Llama 3.2 90B Vision"
        },
        "Microsoft": {
            "microsoft/phi-4-multimodal-instruct": "Phi-4 Multimodal"
        },
        "Mistral": {
            "mistralai/pixtral-12b": "Pixtral 12B",
            "mistralai/pixtral-large-2411": "Pixtral Large"
        },
        "Others": {
            "qwen/qwen-2-vl-72b-instruct": "Qwen2-VL 72B",
            "qwen/qwen-2-vl-7b-instruct": "Qwen2-VL 7B"
        }
    })
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Singleton config instance
config = OCRConfig()