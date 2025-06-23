import httpx
import json
from typing import Optional, Dict, Any
from PIL import Image
import asyncio

from .base import BaseOCRProvider, OCRResult

class OpenRouterProvider(BaseOCRProvider):
    """OpenRouter provider for multiple VLM models"""
    
    def __init__(self, config):
        super().__init__(config)
        self.provider_name = "openrouter"
        self.api_key = config.openrouter_api_key
        self.base_url = config.openrouter_base_url
        
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
            
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def extract_text(
        self, 
        image: Image.Image, 
        model: str = "gpt-4o",
        prompt: Optional[str] = None
    ) -> OCRResult:
        """Extract text using OpenRouter API"""
        
        if not prompt:
            prompt = self.config.default_ocr_prompt
            
        # Get model configuration
        model_config = self.config.available_models.get(model, {})
        
        try:
            # Convert image to base64
            image_b64 = self.image_to_base64(image)
            
            # Prepare request payload
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 4000,
                "temperature": 0.1
            }
            
            # Make API request with timeout and retries
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                for attempt in range(self.config.max_retries):
                    try:
                        response = await client.post(
                            f"{self.base_url}/chat/completions",
                            headers=self.headers,
                            json=payload
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            # Extract text from response
                            text = data['choices'][0]['message']['content'].strip()
                            
                            # Calculate estimated cost
                            usage = data.get('usage', {})
                            total_tokens = usage.get('total_tokens', len(text) // 4)
                            cost = self.estimate_cost(len(text), model_config)
                            
                            return OCRResult(
                                text=text,
                                execution_time=0,  # Will be set by decorator
                                confidence=0.9,  # VLMs don't provide confidence scores
                                provider=f"openrouter/{model_config.get('provider', 'unknown')}",
                                model=model,
                                cost=cost,
                                metadata={
                                    'tokens_used': total_tokens,
                                    'model_config': model_config,
                                    'attempt': attempt + 1
                                }
                            )
                        else:
                            error_msg = f"API Error {response.status_code}: {response.text}"
                            if attempt == self.config.max_retries - 1:
                                return OCRResult(
                                    text="",
                                    execution_time=0,
                                    error=error_msg,
                                    provider=f"openrouter/{model_config.get('provider', 'unknown')}",
                                    model=model
                                )
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            
                    except httpx.TimeoutException:
                        if attempt == self.config.max_retries - 1:
                            return OCRResult(
                                text="",
                                execution_time=0,
                                error="Request timeout",
                                provider=f"openrouter/{model_config.get('provider', 'unknown')}",
                                model=model
                            )
                        await asyncio.sleep(2 ** attempt)
                        
                    except Exception as e:
                        if attempt == self.config.max_retries - 1:
                            return OCRResult(
                                text="",
                                execution_time=0,
                                error=str(e),
                                provider=f"openrouter/{model_config.get('provider', 'unknown')}",
                                model=model
                            )
                        await asyncio.sleep(2 ** attempt)
                        
        except Exception as e:
            return OCRResult(
                text="",
                execution_time=0,
                error=f"Provider error: {str(e)}",
                provider=f"openrouter/{model_config.get('provider', 'unknown')}",
                model=model
            )
    
    async def batch_extract_text(
        self, 
        images: list[Image.Image], 
        model: str = "gpt-4o",
        prompt: Optional[str] = None
    ) -> list[OCRResult]:
        """Extract text from multiple images concurrently"""
        
        # Create tasks for concurrent processing
        tasks = [
            self.extract_text(image, model, prompt) 
            for image in images
        ]
        
        # Execute tasks concurrently
        return await asyncio.gather(*tasks)
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models"""
        return self.config.available_models