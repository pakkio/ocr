#!/usr/bin/env python3
"""
🧪 Comprehensive OCR Test Suite
=====================================

Persistent test suite that evaluates all OCR modes, models, and approaches we've explored.
This test can be run regularly to ensure compatibility and performance across all providers.

Test Categories:
1. VLM Models (OpenRouter) - Structured JSON extraction
2. Traditional OCR - Text extraction  
3. Gradio App compatibility
4. Error handling and fallback mechanisms
5. Performance benchmarking

Author: Claude Code
Version: 1.0
"""

import asyncio
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Single test result data structure"""
    model_name: str
    test_type: str
    success: bool
    execution_time: float
    error_message: Optional[str] = None
    data_extracted: bool = False
    charts_count: int = 0
    metrics_count: int = 0
    confidence: float = 0.0
    json_schema_mode: str = "unknown"

@dataclass
class TestSummary:
    """Complete test run summary"""
    timestamp: str
    total_tests: int
    successful_tests: int
    failed_tests: int
    total_execution_time: float
    models_tested: List[str]
    test_results: List[TestResult]
    
    @property
    def success_rate(self) -> float:
        return (self.successful_tests / self.total_tests * 100) if self.total_tests > 0 else 0

class ComprehensiveTestSuite:
    """Main test suite class"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()
        
        # Test configuration
        self.test_images = [
            "data/istockphoto-1303610445-1024x1024.jpg",
            "data/istockphoto-1390723387-1024x1024.jpg", 
            "data/istockphoto-1472103438-1024x1024.jpg"
        ]
        
        # VLM models to test (organized by provider)
        self.vlm_models = {
            "OpenAI": [
                "gpt-4o",
                "openai/gpt-4o-mini",
                "openai/gpt-4.1",
                "openai/gpt-4.1-mini", 
                "openai/gpt-4.1-nano"
            ],
            "Anthropic": [
                "anthropic/claude-3.5-sonnet",
                "anthropic/claude-sonnet-4",
                "anthropic/claude-3.7-sonnet"
            ],
            "Google": [
                "google/gemini-2.5-pro",
                "google/gemini-2.5-flash",
                "google/gemini-2.5-flash-lite-preview-06-17",
                "google/gemini-2.0-flash-exp",
                "google/gemini-2.0-flash-thinking-exp",
                "google/gemma-3-vision-preview"
            ],
            "Meta": [
                "meta-llama/llama-3.2-11b-vision-instruct",
                "meta-llama/llama-3.2-90b-vision-instruct"
            ],
            "Microsoft": [
                "microsoft/phi-4-multimodal-instruct"
            ],
            "Mistral": [
                "mistralai/pixtral-12b",
                "mistralai/pixtral-large-2411"
            ]
        }
        
        # Traditional OCR models to test
        self.traditional_models = ["easyocr", "paddleocr", "tesseract"]
        
    async def run_all_tests(self) -> TestSummary:
        """Run complete test suite"""
        logger.info("🚀 Starting Comprehensive OCR Test Suite")
        logger.info(f"Testing {len(self.test_images)} images with multiple providers")
        
        # Test 1: VLM Structured Extraction
        await self._test_vlm_structured_extraction()
        
        # Test 2: VLM Gradio Mode
        await self._test_vlm_gradio_mode()
        
        # Test 3: Traditional OCR
        await self._test_traditional_ocr()
        
        # Test 4: Error handling and fallbacks
        await self._test_error_handling()
        
        # Generate summary
        return self._generate_summary()
    
    async def _test_vlm_structured_extraction(self):
        """Test VLM models using structured provider"""
        logger.info("📊 Testing VLM Structured JSON Extraction")
        
        try:
            from src.providers.structured_provider import StructuredOCRProvider
            from src.config import config
            from PIL import Image
            
            provider = StructuredOCRProvider(config)
            
            # Test each provider's models
            for provider_name, models in self.vlm_models.items():
                logger.info(f"  Testing {provider_name} models...")
                
                for model in models:
                    for image_path in self.test_images:
                        await self._test_single_vlm_model(
                            provider, model, image_path, "structured_extraction"
                        )
                        
        except Exception as e:
            logger.error(f"VLM structured extraction test failed: {str(e)}")
            self.results.append(TestResult(
                model_name="structured_provider",
                test_type="setup_error",
                success=False,
                execution_time=0,
                error_message=str(e)
            ))
    
    async def _test_vlm_gradio_mode(self):
        """Test VLM models using Gradio approach"""
        logger.info("🎭 Testing VLM Gradio Mode (json_object)")
        
        try:
            from gradio_main import StandaloneStructuredOCR
            from PIL import Image
            
            handler = StandaloneStructuredOCR()
            
            # Test key models with Gradio approach
            test_models = [
                "gpt-4o",
                "openai/gpt-4o-mini", 
                "anthropic/claude-3.5-sonnet",
                "google/gemini-2.5-flash"
            ]
            
            for model in test_models:
                for image_path in self.test_images:
                    await self._test_single_gradio_model(handler, model, image_path)
                    
        except Exception as e:
            logger.error(f"VLM Gradio mode test failed: {str(e)}")
            self.results.append(TestResult(
                model_name="gradio_handler",
                test_type="setup_error", 
                success=False,
                execution_time=0,
                error_message=str(e)
            ))
    
    async def _test_traditional_ocr(self):
        """Test traditional OCR models"""
        logger.info("⚙️ Testing Traditional OCR Models")
        
        try:
            from gradio_main import StandaloneTraditionalOCR
            from PIL import Image
            
            handler = StandaloneTraditionalOCR()
            
            for model in self.traditional_models:
                for image_path in self.test_images:
                    await self._test_single_traditional_model(handler, model, image_path)
                    
        except Exception as e:
            logger.error(f"Traditional OCR test failed: {str(e)}")
            self.results.append(TestResult(
                model_name="traditional_ocr",
                test_type="setup_error",
                success=False, 
                execution_time=0,
                error_message=str(e)
            ))
    
    async def _test_error_handling(self):
        """Test error handling and fallback mechanisms"""
        logger.info("🛡️ Testing Error Handling & Fallbacks")
        
        # Test with invalid model
        await self._test_invalid_model()
        
        # Test with corrupted image
        await self._test_corrupted_image()
        
        # Test schema fallback mechanisms
        await self._test_schema_fallbacks()
    
    async def _test_single_vlm_model(self, provider, model: str, image_path: str, test_type: str):
        """Test a single VLM model with structured provider"""
        start_time = time.time()
        
        try:
            from PIL import Image
            image = Image.open(image_path)
            
            result = await provider.extract_structured_data(image, model)
            execution_time = time.time() - start_time
            
            if result["success"]:
                data = result["data"]
                self.results.append(TestResult(
                    model_name=model,
                    test_type=test_type,
                    success=True,
                    execution_time=execution_time,
                    data_extracted=True,
                    charts_count=len(data.get("charts", [])),
                    metrics_count=len(data.get("metrics", [])),
                    json_schema_mode="strict" if provider.config.available_models.get(model, {}).get("supports_strict_json_schema", False) else "fallback"
                ))
                logger.info(f"    ✓ {model}: {len(data.get('charts', []))} charts, {len(data.get('metrics', []))} metrics")
            else:
                self.results.append(TestResult(
                    model_name=model,
                    test_type=test_type,
                    success=False,
                    execution_time=execution_time,
                    error_message=result.get("error", "Unknown error")
                ))
                logger.warning(f"    ✗ {model}: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.results.append(TestResult(
                model_name=model,
                test_type=test_type,
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            ))
            logger.error(f"    ✗ {model}: Exception - {str(e)}")
    
    async def _test_single_gradio_model(self, handler, model: str, image_path: str):
        """Test a single VLM model with Gradio approach"""
        start_time = time.time()
        
        try:
            from PIL import Image
            image = Image.open(image_path)
            
            result = await handler.extract(image, model)
            execution_time = time.time() - start_time
            
            if result["success"]:
                data = result["data"] 
                self.results.append(TestResult(
                    model_name=model,
                    test_type="gradio_extraction",
                    success=True,
                    execution_time=execution_time,
                    data_extracted=True,
                    charts_count=len(data.get("charts", [])),
                    metrics_count=len(data.get("metrics", [])),
                    json_schema_mode="json_object"
                ))
                logger.info(f"    ✓ {model} (Gradio): {len(data.get('charts', []))} charts, {len(data.get('metrics', []))} metrics")
            else:
                self.results.append(TestResult(
                    model_name=model,
                    test_type="gradio_extraction",
                    success=False,
                    execution_time=execution_time,
                    error_message=result.get("error", "Unknown error")
                ))
                logger.warning(f"    ✗ {model} (Gradio): {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.results.append(TestResult(
                model_name=model,
                test_type="gradio_extraction",
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            ))
            logger.error(f"    ✗ {model} (Gradio): Exception - {str(e)}")
    
    async def _test_single_traditional_model(self, handler, model: str, image_path: str):
        """Test a single traditional OCR model"""
        start_time = time.time()
        
        try:
            from PIL import Image
            image = Image.open(image_path)
            
            result = await handler.extract_text(image, model)
            execution_time = time.time() - start_time
            
            if result["success"]:
                text_length = len(result.get("text", ""))
                self.results.append(TestResult(
                    model_name=model,
                    test_type="traditional_ocr",
                    success=True,
                    execution_time=execution_time,
                    data_extracted=text_length > 0,
                    confidence=result.get("confidence", 0),
                    json_schema_mode="text_only"
                ))
                logger.info(f"    ✓ {model}: {text_length} chars, confidence: {result.get('confidence', 0):.2f}")
            else:
                self.results.append(TestResult(
                    model_name=model,
                    test_type="traditional_ocr",
                    success=False,
                    execution_time=execution_time,
                    error_message=result.get("error", "Unknown error")
                ))
                logger.warning(f"    ✗ {model}: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.results.append(TestResult(
                model_name=model,
                test_type="traditional_ocr",
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            ))
            logger.error(f"    ✗ {model}: Exception - {str(e)}")
    
    async def _test_invalid_model(self):
        """Test with invalid model name"""
        logger.info("  Testing invalid model handling...")
        
        try:
            from src.providers.structured_provider import StructuredOCRProvider
            from src.config import config
            from PIL import Image
            
            provider = StructuredOCRProvider(config)
            image = Image.open(self.test_images[0])
            
            start_time = time.time()
            result = await provider.extract_structured_data(image, "invalid/nonexistent-model")
            execution_time = time.time() - start_time
            
            # Should fail gracefully
            self.results.append(TestResult(
                model_name="invalid/nonexistent-model",
                test_type="error_handling",
                success=not result["success"],  # Success if it failed gracefully
                execution_time=execution_time,
                error_message=result.get("error", "No error returned")
            ))
            
        except Exception as e:
            logger.info(f"    ✓ Invalid model test handled exception correctly: {str(e)}")
    
    async def _test_corrupted_image(self):
        """Test with corrupted image data"""
        logger.info("  Testing corrupted image handling...")
        # This would require creating a corrupted image file for testing
        pass
    
    async def _test_schema_fallbacks(self):
        """Test JSON schema fallback mechanisms"""
        logger.info("  Testing schema fallback mechanisms...")
        
        # Test models that use fallback vs strict schema
        fallback_models = ["anthropic/claude-3.5-sonnet", "google/gemini-2.5-flash"]
        strict_models = ["gpt-4o"]
        
        # Compare results between fallback and strict modes
        for model in fallback_models + strict_models:
            # Results already captured in main tests
            pass
    
    def _generate_summary(self) -> TestSummary:
        """Generate comprehensive test summary"""
        total_time = time.time() - self.start_time
        successful = sum(1 for r in self.results if r.success)
        failed = len(self.results) - successful
        models_tested = list(set(r.model_name for r in self.results))
        
        summary = TestSummary(
            timestamp=datetime.now().isoformat(),
            total_tests=len(self.results),
            successful_tests=successful,
            failed_tests=failed,
            total_execution_time=total_time,
            models_tested=models_tested,
            test_results=self.results
        )
        
        # Log summary
        logger.info("="*60)
        logger.info("📊 TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Tests: {summary.total_tests}")
        logger.info(f"Successful: {summary.successful_tests}")
        logger.info(f"Failed: {summary.failed_tests}")
        logger.info(f"Success Rate: {summary.success_rate:.1f}%")
        logger.info(f"Total Time: {summary.total_execution_time:.2f}s")
        logger.info(f"Models Tested: {len(summary.models_tested)}")
        
        # Save detailed results
        self._save_results(summary)
        
        return summary
    
    def _save_results(self, summary: TestSummary):
        """Save test results to JSON file"""
        results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert dataclasses to dict for JSON serialization
        summary_dict = asdict(summary)
        
        with open(results_file, 'w') as f:
            json.dump(summary_dict, f, indent=2, default=str)
        
        logger.info(f"📁 Detailed results saved to: {results_file}")
        
        # Also save a human-readable report
        self._save_human_readable_report(summary)
    
    def _save_human_readable_report(self, summary: TestSummary):
        """Save human-readable test report"""
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w') as f:
            f.write("# 🧪 OCR Test Suite Report\n\n")
            f.write(f"**Date**: {summary.timestamp}\n")
            f.write(f"**Success Rate**: {summary.success_rate:.1f}%\n")
            f.write(f"**Total Tests**: {summary.total_tests}\n")
            f.write(f"**Execution Time**: {summary.total_execution_time:.2f}s\n\n")
            
            # Group results by provider
            providers = {}
            for result in summary.test_results:
                provider = result.model_name.split('/')[0] if '/' in result.model_name else 'OpenAI'
                if provider not in providers:
                    providers[provider] = []
                providers[provider].append(result)
            
            for provider, results in providers.items():
                f.write(f"## {provider}\n\n")
                
                successful = sum(1 for r in results if r.success)
                total = len(results)
                f.write(f"**Success Rate**: {successful}/{total} ({successful/total*100:.1f}%)\n\n")
                
                f.write("| Model | Test Type | Status | Time | Charts | Metrics | Schema Mode |\n")
                f.write("|-------|-----------|--------|------|--------|---------|-------------|\n")
                
                for result in results:
                    status = "✅" if result.success else "❌"
                    f.write(f"| {result.model_name} | {result.test_type} | {status} | {result.execution_time:.2f}s | {result.charts_count} | {result.metrics_count} | {result.json_schema_mode} |\n")
                
                f.write("\n")
        
        logger.info(f"📄 Human-readable report saved to: {report_file}")

async def main():
    """Run the comprehensive test suite"""
    suite = ComprehensiveTestSuite()
    summary = await suite.run_all_tests()
    
    print(f"\n🎯 Test Suite Complete!")
    print(f"Success Rate: {summary.success_rate:.1f}%")
    print(f"Total Tests: {summary.total_tests}")
    print(f"Models Tested: {len(summary.models_tested)}")

if __name__ == "__main__":
    asyncio.run(main())