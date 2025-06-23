from typing import Dict, List, Optional, Union
from enum import Enum

from .config import OCRConfig
from .providers.base import BaseOCRProvider
from .providers.openrouter_provider import OpenRouterProvider
from .providers.traditional_provider import TraditionalOCRProvider

class ProviderType(Enum):
    OPENROUTER = "openrouter"
    TRADITIONAL = "traditional"

class OCRProviderFactory:
    """Factory class for creating OCR providers with dependency injection"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self._providers: Dict[str, BaseOCRProvider] = {}
    
    def create_provider(self, provider_type: Union[ProviderType, str]) -> BaseOCRProvider:
        """Create and cache OCR provider instances"""
        
        if isinstance(provider_type, str):
            provider_type = ProviderType(provider_type)
        
        provider_key = provider_type.value
        
        # Return cached provider if available
        if provider_key in self._providers:
            return self._providers[provider_key]
        
        # Create new provider based on type
        if provider_type == ProviderType.OPENROUTER:
            provider = OpenRouterProvider(self.config)
        elif provider_type == ProviderType.TRADITIONAL:
            provider = TraditionalOCRProvider(self.config)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
        
        # Cache and return
        self._providers[provider_key] = provider
        return provider
    
    def get_available_models(self) -> Dict[str, Dict]:
        """Get all available models from all providers"""
        models = {}
        
        # VLM models from OpenRouter
        if self.config.openrouter_api_key:
            models.update({
                f"vlm_{model_id}": {
                    **model_config,
                    "provider_type": "openrouter",
                    "model_id": model_id
                }
                for model_id, model_config in self.config.available_models.items()
            })
        
        # Traditional OCR models
        traditional_models = TraditionalOCRProvider.get_available_models()
        models.update({
            f"traditional_{model_id}": {
                **model_config,
                "provider_type": "traditional", 
                "model_id": model_id,
                "cost_per_1k_tokens": 0.0  # Traditional OCR is free
            }
            for model_id, model_config in traditional_models.items()
            if model_config.get("available", False)
        })
        
        return models
    
    def create_benchmark_suite(self, selected_models: Optional[List[str]] = None) -> List[tuple]:
        """Create a benchmark suite with provider and model combinations"""
        available_models = self.get_available_models()
        
        if selected_models:
            # Filter to selected models only
            available_models = {
                k: v for k, v in available_models.items()
                if k in selected_models or v.get("model_id") in selected_models
            }
        
        benchmark_suite = []
        
        for full_model_id, model_config in available_models.items():
            provider_type = model_config["provider_type"]
            model_id = model_config["model_id"]
            
            provider = self.create_provider(provider_type)
            benchmark_suite.append((provider, model_id, model_config))
        
        return benchmark_suite

class DIContainer:
    """Simple Dependency Injection Container"""
    
    def __init__(self):
        self._services = {}
        self._singletons = {}
    
    def register(self, service_type: type, implementation: type, singleton: bool = True):
        """Register a service implementation"""
        self._services[service_type] = implementation
        if singleton:
            self._singletons[service_type] = None
    
    def get(self, service_type: type, *args, **kwargs):
        """Get service instance"""
        if service_type not in self._services:
            raise ValueError(f"Service {service_type} not registered")
        
        # Return singleton if cached
        if service_type in self._singletons:
            if self._singletons[service_type] is None:
                self._singletons[service_type] = self._services[service_type](*args, **kwargs)
            return self._singletons[service_type]
        
        # Return new instance
        return self._services[service_type](*args, **kwargs)

# Setup DI Container
def setup_di_container() -> DIContainer:
    """Setup dependency injection container with default services"""
    container = DIContainer()
    
    # Register core services
    container.register(OCRConfig, OCRConfig, singleton=True)
    container.register(OCRProviderFactory, OCRProviderFactory, singleton=True)
    
    return container

# Global DI container instance
di_container = setup_di_container()