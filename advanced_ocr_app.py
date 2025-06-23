# Advanced OCR Benchmark - Standalone Class (converted from Streamlit)
import asyncio
import pandas as pd
import numpy as np
from PIL import Image
import io
import zipfile
import json
from typing import List, Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Import our custom modules
from src.config import config
from src.factory import OCRProviderFactory, di_container
from src.providers.base import OCRResult

class AdvancedOCRBenchmark:
    """Advanced OCR benchmarking application"""
    
    def __init__(self):
        self.config = config
        self.factory = OCRProviderFactory(self.config)
        self.results_history = []
    
    async def run_single_benchmark(
        self, 
        image: Image.Image, 
        provider, 
        model_id: str, 
        model_config: Dict,
        custom_prompt: str = None
    ) -> OCRResult:
        """Run OCR on single image with single model"""
        
        prompt = custom_prompt or self.config.default_ocr_prompt
        
        try:
            if hasattr(provider, 'extract_text'):
                if "traditional" in str(type(provider)):
                    result = await provider.extract_text(image, model_id)
                else:
                    result = await provider.extract_text(image, model_id, prompt)
                return result
            else:
                return OCRResult(
                    text="",
                    execution_time=0,
                    error="Provider doesn't support extract_text method",
                    provider=str(type(provider)),
                    model=model_id
                )
        except Exception as e:
            return OCRResult(
                text="",
                execution_time=0,
                error=str(e),
                provider=str(type(provider)),
                model=model_id
            )
    
    async def run_batch_benchmark(
        self, 
        images: List[Image.Image], 
        selected_models: List[str],
        custom_prompt: str = None
    ) -> List[Dict[str, Any]]:
        """Run comprehensive benchmark on multiple images and models"""
        
        benchmark_suite = self.factory.create_benchmark_suite(selected_models)
        all_results = []
        
        for i, image in enumerate(images):
            print(f"🔄 Processing image {i+1}/{len(images)}...")
            
            for j, (provider, model_id, model_config) in enumerate(benchmark_suite):
                # Update progress
                progress = (j + 1) / len(benchmark_suite)
                print(f"   Progress: {progress*100:.1f}% - Testing: {model_config.get('name', model_id)}")
                
                # Run OCR
                result = await self.run_single_benchmark(
                    image, provider, model_id, model_config, custom_prompt
                )
                
                # Store result with metadata
                result_data = result.to_dict()
                result_data.update({
                    'image_index': i,
                    'image_name': f"image_{i+1}",
                    'timestamp': datetime.now().isoformat(),
                    'model_name': model_config.get('name', model_id)
                })
                
                all_results.append(result_data)
            
            print(f"✅ Completed image {i+1}/{len(images)}")
        
        return all_results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze benchmark results and generate insights"""
        
        df = pd.DataFrame(results)
        
        if df.empty:
            return {"error": "No results to analyze"}
        
        analysis = {
            "summary": {
                "total_tests": len(df),
                "unique_models": df['model'].nunique(),
                "unique_images": df['image_index'].nunique(),
                "success_rate": (df['error'].isna().sum() / len(df)) * 100,
                "average_execution_time": df[df['error'].isna()]['execution_time'].mean(),
                "total_cost": df[df['error'].isna()]['cost'].sum()
            },
            "model_performance": {},
            "image_difficulty": {},
            "cost_analysis": {}
        }
        
        # Model performance analysis
        model_stats = df.groupby('model').agg({
            'execution_time': ['mean', 'std'],
            'confidence': ['mean', 'std'],
            'character_count': ['mean', 'std'],
            'cost': 'sum',
            'error': lambda x: x.isna().sum() / len(x) * 100  # success rate
        }).round(4)
        
        analysis["model_performance"] = model_stats.to_dict()
        
        # Image difficulty analysis (based on processing time and success rates)
        image_stats = df.groupby('image_index').agg({
            'execution_time': 'mean',
            'character_count': 'mean',
            'error': lambda x: x.isna().sum() / len(x) * 100
        }).round(4)
        
        analysis["image_difficulty"] = image_stats.to_dict()
        
        # Cost analysis
        cost_by_provider = df.groupby('provider')['cost'].sum().to_dict()
        analysis["cost_analysis"] = cost_by_provider
        
        return analysis

def main():
    """Standalone CLI version - use gradio_main.py for web interface"""
    print("🚀 Advanced OCR Benchmark Suite - Standalone Mode")
    print("Note: Use gradio_main.py for web interface")
    
    # Check API key configuration
    if not config.openrouter_api_key:
        print("⚠️ OpenRouter API key not configured. Create `.env` file with OPENROUTER_API_KEY")
        print("Copy `.env.example` to `.env` and add your API key")
        return
    
    # Initialize benchmark app
    benchmark_app = AdvancedOCRBenchmark()
    print(f"✅ Initialized with {len(benchmark_app.config.available_models)} available models")
    
    # Example usage - for CLI testing
    print("\nTo use this class in your own code:")
    print("from advanced_ocr_app import AdvancedOCRBenchmark, run_batch_benchmark_cli")

# Utility functions for external usage
def run_batch_benchmark_cli(images: List[Image.Image], selected_models: List[str], custom_prompt: str = None):
    """CLI wrapper for batch benchmark"""
    if not config.openrouter_api_key:
        print("❌ OpenRouter API key not configured")
        return None
    
    benchmark_app = AdvancedOCRBenchmark()
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(
            benchmark_app.run_batch_benchmark(
                images, selected_models, custom_prompt
            )
        )
        
        loop.close()
        return results
        
    except Exception as e:
        print(f"❌ Benchmark failed: {str(e)}")
        return None

def export_results_to_json(results: List[Dict[str, Any]], filename: str = None):
    """Export results to JSON file"""
    if not filename:
        filename = f"ocr_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✅ Results exported to {filename}")

if __name__ == "__main__":
    main()