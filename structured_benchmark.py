"""
Structured Benchmark - Standalone Class for VLM Structured Extraction
=====================================================================

Pure Python class for benchmarking Vision Language Models on structured
JSON extraction from dashboard images with quality assessment.

Usage:
    benchmark = StructuredBenchmark()
    results = benchmark.run_benchmark(image_files, models)
    
Note: UI functionality is available in gradio_main.py
"""

import asyncio
import pandas as pd
import json
import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
from datetime import datetime

try:
    from src.providers.structured_provider import StructuredOCRProvider
    from src.config import config
    from src.judge_llm import JudgeLLM, judge_ocr_results
    STRUCTURED_PROVIDER_AVAILABLE = True
except ImportError:
    STRUCTURED_PROVIDER_AVAILABLE = False
    print("Warning: Structured provider not available - check src/ directory")

class StructuredBenchmark:
    """Standalone class for VLM structured extraction benchmarking"""
    
    def __init__(self):
        """Initialize structured provider and judge LLM with error handling"""
        self.provider = None
        self.judge = None
        if STRUCTURED_PROVIDER_AVAILABLE:
            try:
                if not config.openrouter_api_key:
                    print("⚠️ OpenRouter API key not configured!")
                    print("Please set OPENROUTER_API_KEY in your .env file")
                else:
                    self.provider = StructuredOCRProvider(config)
                    self.judge = JudgeLLM()
            except Exception as e:
                print(f"❌ Failed to initialize structured provider: {str(e)}")

    def discover_images(self, data_dir: str = "data") -> List[str]:
        """Auto-discover images in data directory"""
        if not self.provider:
            return []
        return self.provider.discover_data_files(data_dir)

    async def extract_structured_data(self, image_path: str, model: str) -> Dict[str, Any]:
        """Extract structured data from single image with single model"""
        if not self.provider:
            return {
                "success": False,
                "error": "Structured provider not available",
                "data": None
            }
        
        try:
            image = Image.open(image_path)
            result = await self.provider.extract_structured_data(image, model)
            return result
        except Exception as e:
            return {
                "success": False,
                "error": f"Image processing failed: {str(e)}",
                "data": None
            }

    async def assess_quality(self, extracted_data: Dict[str, Any], image_description: str) -> Dict[str, Any]:
        """Assess quality of extracted data"""
        if not self.provider:
            return {
                "success": False,
                "error": "Structured provider not available",
                "assessment": None
            }
        
        try:
            result = await self.provider.assess_extraction_quality(extracted_data, image_description)
            return result
        except Exception as e:
            return {
                "success": False,
                "error": f"Quality assessment failed: {str(e)}",
                "assessment": None
            }

    async def run_single_benchmark(self, image_path: str, models: List[str]) -> Dict[str, Any]:
        """Run benchmark on single image with multiple models"""
        image_name = os.path.basename(image_path)
        results = []
        
        for model in models:
            print(f"  Testing {model} on {image_name}...")
            
            # Extract structured data
            extraction_result = await self.extract_structured_data(image_path, model)
            
            result = {
                "image_path": image_path,
                "image_name": image_name,
                "model": model,
                "extraction_result": extraction_result
            }
            
            # If extraction succeeded, assess quality
            if extraction_result.get("success"):
                image_description = f"Dashboard image: {image_name}"
                quality_result = await self.assess_quality(
                    extraction_result["data"],
                    image_description
                )
                result["quality_assessment"] = quality_result
            else:
                result["quality_assessment"] = {
                    "success": False,
                    "error": "Extraction failed, cannot assess quality"
                }
            
            results.append(result)
        
        return {
            "image_path": image_path,
            "image_name": image_name,
            "results": results
        }

    async def run_benchmark(self, image_files: Optional[List[str]] = None, 
                           models: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run comprehensive benchmark on multiple images and models
        
        Args:
            image_files: List of image file paths. If None, auto-discovers from data/
            models: List of model names. If None, uses default set
            
        Returns:
            Dictionary with complete benchmark results
        """
        if not self.provider:
            return {
                "error": "Structured provider not available",
                "results": []
            }
        
        # Auto-discover images if not provided
        if image_files is None:
            image_files = self.discover_images()
            
        if not image_files:
            return {
                "error": "No images found for testing",
                "results": []
            }
        
        # Use default models if not provided
        if models is None:
            models = ["gpt-4o", "anthropic/claude-3.5-sonnet", "google/gemini-2.5-flash"]
        
        print(f"🚀 Starting benchmark with {len(image_files)} images and {len(models)} models")
        
        # Run benchmark for each image
        all_results = []
        for image_path in image_files:
            image_result = await self.run_single_benchmark(image_path, models)
            all_results.append(image_result)
        
        # Compile summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_images": len(image_files),
            "models_tested": models,
            "total_tests": len(image_files) * len(models),
            "results": all_results
        }
        
        return summary

    def analyze_results(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze benchmark results and provide insights"""
        if "error" in benchmark_results:
            return {"error": benchmark_results["error"]}
        
        analysis = {
            "summary": {
                "total_tests": benchmark_results.get("total_tests", 0),
                "images_processed": len(benchmark_results.get("results", [])),
                "models_tested": len(benchmark_results.get("models_tested", []))
            },
            "success_rates": {},
            "performance_metrics": {},
            "quality_scores": {}
        }
        
        # Analyze by model
        model_stats = {}
        
        for image_result in benchmark_results.get("results", []):
            for test_result in image_result.get("results", []):
                model = test_result["model"]
                
                if model not in model_stats:
                    model_stats[model] = {
                        "total_tests": 0,
                        "successful_extractions": 0,
                        "quality_assessments": 0,
                        "total_charts": 0,
                        "total_metrics": 0,
                        "execution_times": []
                    }
                
                stats = model_stats[model]
                stats["total_tests"] += 1
                
                extraction = test_result.get("extraction_result", {})
                if extraction.get("success"):
                    stats["successful_extractions"] += 1
                    
                    # Count extracted data
                    data = extraction.get("data", {})
                    stats["total_charts"] += len(data.get("charts", []))
                    stats["total_metrics"] += len(data.get("metrics", []))
                
                quality = test_result.get("quality_assessment", {})
                if quality.get("success"):
                    stats["quality_assessments"] += 1
        
        # Calculate metrics
        for model, stats in model_stats.items():
            if stats["total_tests"] > 0:
                analysis["success_rates"][model] = {
                    "extraction_rate": stats["successful_extractions"] / stats["total_tests"],
                    "quality_assessment_rate": stats["quality_assessments"] / stats["total_tests"]
                }
                
                analysis["performance_metrics"][model] = {
                    "avg_charts_per_image": stats["total_charts"] / stats["total_tests"],
                    "avg_metrics_per_image": stats["total_metrics"] / stats["total_tests"]
                }
        
        return analysis

    async def run_judge_comparison(self, result_a: Dict[str, Any], result_b: Dict[str, Any], 
                                 image_path: str, model_a_name: str, model_b_name: str) -> Dict[str, Any]:
        """
        Compare two OCR results using the judge LLM
        
        Args:
            result_a: First OCR extraction result
            result_b: Second OCR extraction result  
            image_path: Path to source image
            model_a_name: Name of first model
            model_b_name: Name of second model
            
        Returns:
            Dictionary with judgment results and human-readable report
        """
        if not self.judge:
            return {
                "success": False,
                "error": "Judge LLM not available",
                "judgment": None,
                "report": None
            }
        
        try:
            # Extract data from extraction results
            data_a = result_a.get("data") if result_a.get("success") else {}
            data_b = result_b.get("data") if result_b.get("success") else {}
            
            # Perform judgment
            judgment = await self.judge.judge_comparison(data_a, data_b, image_path)
            
            # Generate human-readable report
            report = self.judge.create_human_readable_report(judgment, model_a_name, model_b_name)
            
            return {
                "success": True,
                "judgment": judgment.model_dump(),
                "report": report
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Judge comparison failed: {str(e)}",
                "judgment": None,
                "report": None
            }

    async def run_judge_tournament(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a tournament-style comparison between all model pairs using judge LLM
        
        Args:
            benchmark_results: Results from run_benchmark()
            
        Returns:
            Dictionary with all pairwise comparisons and rankings
        """
        if not self.judge:
            return {
                "error": "Judge LLM not available",
                "comparisons": [],
                "rankings": {}
            }
        
        if "error" in benchmark_results:
            return {
                "error": benchmark_results["error"],
                "comparisons": [],
                "rankings": {}
            }
        
        print("🏆 Starting judge tournament...")
        
        tournament_results = {
            "timestamp": datetime.now().isoformat(),
            "comparisons": [],
            "rankings": {},
            "summary": {}
        }
        
        # Extract all model results per image
        for image_result in benchmark_results.get("results", []):
            image_path = image_result["image_path"]
            image_name = image_result["image_name"]
            
            print(f"  Judging {image_name}...")
            
            model_results = {}
            for test_result in image_result.get("results", []):
                model = test_result["model"]
                model_results[model] = test_result["extraction_result"]
            
            # Compare all pairs of models for this image
            models = list(model_results.keys())
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    model_a = models[i]
                    model_b = models[j]
                    
                    print(f"    {model_a} vs {model_b}")
                    
                    comparison = await self.run_judge_comparison(
                        model_results[model_a],
                        model_results[model_b],
                        image_path,
                        model_a,
                        model_b
                    )
                    
                    comparison_record = {
                        "image_path": image_path,
                        "image_name": image_name,
                        "model_a": model_a,
                        "model_b": model_b,
                        "comparison": comparison
                    }
                    
                    tournament_results["comparisons"].append(comparison_record)
        
        # Calculate overall rankings
        model_scores = {}
        win_matrix = {}
        
        for comparison in tournament_results["comparisons"]:
            if not comparison["comparison"]["success"]:
                continue
                
            judgment = comparison["comparison"]["judgment"]
            model_a = comparison["model_a"]
            model_b = comparison["model_b"]
            winner = judgment.get("winner")
            
            # Initialize score tracking
            if model_a not in model_scores:
                model_scores[model_a] = {"wins": 0, "losses": 0, "ties": 0, "total_score": 0.0}
            if model_b not in model_scores:
                model_scores[model_b] = {"wins": 0, "losses": 0, "ties": 0, "total_score": 0.0}
            
            # Initialize win matrix
            if model_a not in win_matrix:
                win_matrix[model_a] = {}
            if model_b not in win_matrix:
                win_matrix[model_b] = {}
            if model_b not in win_matrix[model_a]:
                win_matrix[model_a][model_b] = {"wins": 0, "losses": 0, "ties": 0}
            if model_a not in win_matrix[model_b]:
                win_matrix[model_b][model_a] = {"wins": 0, "losses": 0, "ties": 0}
            
            # Update scores based on judgment
            if winner == "result_a":
                model_scores[model_a]["wins"] += 1
                model_scores[model_b]["losses"] += 1
                win_matrix[model_a][model_b]["wins"] += 1
                win_matrix[model_b][model_a]["losses"] += 1
            elif winner == "result_b":
                model_scores[model_b]["wins"] += 1
                model_scores[model_a]["losses"] += 1
                win_matrix[model_b][model_a]["wins"] += 1
                win_matrix[model_a][model_b]["losses"] += 1
            else:  # tie
                model_scores[model_a]["ties"] += 1
                model_scores[model_b]["ties"] += 1
                win_matrix[model_a][model_b]["ties"] += 1
                win_matrix[model_b][model_a]["ties"] += 1
            
            # Add overall scores from judgment
            overall_scores = judgment.get("overall_scores", {})
            model_scores[model_a]["total_score"] += overall_scores.get("result_a", 5.0)
            model_scores[model_b]["total_score"] += overall_scores.get("result_b", 5.0)
        
        # Calculate win rates and average scores
        for model, scores in model_scores.items():
            total_games = scores["wins"] + scores["losses"] + scores["ties"]
            if total_games > 0:
                scores["win_rate"] = scores["wins"] / total_games
                scores["average_score"] = scores["total_score"] / total_games
            else:
                scores["win_rate"] = 0.0
                scores["average_score"] = 0.0
        
        # Rank models by win rate, then by average score
        sorted_models = sorted(
            model_scores.items(),
            key=lambda x: (x[1]["win_rate"], x[1]["average_score"]),
            reverse=True
        )
        
        tournament_results["rankings"] = {
            "model_scores": model_scores,
            "win_matrix": win_matrix,
            "leaderboard": [{"rank": i+1, "model": model, **scores} 
                          for i, (model, scores) in enumerate(sorted_models)]
        }
        
        tournament_results["summary"] = {
            "total_comparisons": len(tournament_results["comparisons"]),
            "successful_judgments": len([c for c in tournament_results["comparisons"] 
                                       if c["comparison"]["success"]]),
            "models_evaluated": len(model_scores),
            "winner": sorted_models[0][0] if sorted_models else None
        }
        
        return tournament_results

    def export_results(self, benchmark_results: Dict[str, Any], 
                      output_dir: str = "results") -> Dict[str, str]:
        """Export benchmark results to files"""
        if "error" in benchmark_results:
            return {"error": benchmark_results["error"]}
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export complete results as JSON
        json_file = os.path.join(output_dir, f"structured_benchmark_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        
        # Export analysis
        analysis = self.analyze_results(benchmark_results)
        analysis_file = os.path.join(output_dir, f"benchmark_analysis_{timestamp}.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Export summary CSV
        csv_data = []
        for image_result in benchmark_results.get("results", []):
            for test_result in image_result.get("results", []):
                extraction = test_result.get("extraction_result", {})
                quality = test_result.get("quality_assessment", {})
                
                row = {
                    "image": test_result["image_name"],
                    "model": test_result["model"],
                    "extraction_success": extraction.get("success", False),
                    "charts_found": len(extraction.get("data", {}).get("charts", [])),
                    "metrics_found": len(extraction.get("data", {}).get("metrics", [])),
                    "quality_success": quality.get("success", False)
                }
                
                if quality.get("success"):
                    assessment = quality.get("assessment", {})
                    row.update({
                        "completeness_score": assessment.get("completeness_score", 0),
                        "accuracy_score": assessment.get("accuracy_score", 0),
                        "structure_score": assessment.get("structure_score", 0),
                        "confidence_level": assessment.get("confidence_level", "unknown")
                    })
                
                csv_data.append(row)
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_file = os.path.join(output_dir, f"benchmark_summary_{timestamp}.csv")
            df.to_csv(csv_file, index=False)
        
        return {
            "json_file": json_file,
            "analysis_file": analysis_file,
            "csv_file": csv_file if csv_data else None
        }

def main():
    """Example usage of StructuredBenchmark"""
    print("🔬 Structured Benchmark - Pure Python VLM Benchmarking")
    print("=" * 60)
    print("This is a standalone class for VLM benchmarking.")
    print("For interactive UI, use: python gradio_main.py")
    print("=" * 60)
    
    if not STRUCTURED_PROVIDER_AVAILABLE:
        print("❌ Structured provider not available")
        print("Please ensure src/ directory is properly configured")
        return
    
    benchmark = StructuredBenchmark()
    
    if not benchmark.provider:
        print("❌ Failed to initialize benchmark")
        print("Please check your OpenRouter API key configuration")
        return
    
    # Discover available images
    images = benchmark.discover_images()
    print(f"📷 Found {len(images)} images in data/ directory:")
    for img in images:
        print(f"  - {os.path.basename(img)}")
    
    print()
    print("To run a benchmark:")
    print("  import asyncio")
    print("  benchmark = StructuredBenchmark()")
    print("  results = asyncio.run(benchmark.run_benchmark())")
    print("  analysis = benchmark.analyze_results(results)")
    print("  files = benchmark.export_results(results)")

if __name__ == "__main__":
    main()