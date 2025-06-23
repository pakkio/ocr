from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from PIL import Image
import time
import base64
import io

class OCRResult:
    """Standard result format for all OCR providers"""
    def __init__(
        self, 
        text: str, 
        execution_time: float, 
        confidence: float = 0.0,
        provider: str = "",
        model: str = "",
        cost: float = 0.0,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.text = text
        self.execution_time = execution_time
        self.confidence = confidence
        self.provider = provider
        self.model = model
        self.cost = cost
        self.error = error
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'execution_time': self.execution_time,
            'confidence': self.confidence,
            'provider': self.provider,
            'model': self.model,
            'cost': self.cost,
            'error': self.error,
            'metadata': self.metadata,
            'character_count': len(self.text),
            'word_count': len(self.text.split()) if self.text else 0
        }

class BaseOCRProvider(ABC):
    """Abstract base class for all OCR providers"""
    
    def __init__(self, config):
        self.config = config
        self.provider_name = "base"
        
    @abstractmethod
    async def extract_text(self, image: Image.Image, prompt: Optional[str] = None) -> OCRResult:
        """Extract text from image"""
        pass
    
    def image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """Convert PIL image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    
    def measure_execution_time(self, func):
        """Decorator to measure execution time"""
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            if hasattr(result, 'execution_time'):
                result.execution_time = execution_time
            return result
        return wrapper
    
    def estimate_cost(self, text_length: int, model_config: Dict[str, Any]) -> float:
        """Estimate cost based on text length and model pricing"""
        # Rough estimation: 1 token ≈ 4 characters
        estimated_tokens = text_length / 4
        cost_per_1k = model_config.get('cost_per_1k_tokens', 0)
        return (estimated_tokens / 1000) * cost_per_1k