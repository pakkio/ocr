#!/usr/bin/env python3
"""
Test Judge System - Demonstration of the judge LLM functionality
=================================================================

This script demonstrates the judge LLM system by:
1. Running OCR extraction on sample images with 2-3 models
2. Using the judge LLM to compare results
3. Running a tournament to rank all models
4. Displaying human-readable comparisons

Usage:
    python test_judge_system.py
"""

import asyncio
import os
from pathlib import Path
from structured_benchmark import StructuredBenchmark


async def test_judge_system():
    """Test the judge LLM system with sample OCR results"""
    print("🧪 Testing Judge LLM System")
    print("=" * 50)
    
    # Initialize benchmark with judge
    benchmark = StructuredBenchmark()
    
    if not benchmark.provider:
        print("❌ Structured provider not available")
        return
    
    if not benchmark.judge:
        print("❌ Judge LLM not available")
        return
    
    # Discover test images
    images = benchmark.discover_images()
    if not images:
        print("❌ No test images found in data/ directory")
        return
    
    print(f"📷 Found {len(images)} test images")
    
    # Use subset of models for testing
    test_models = [
        "gpt-4o",
        "anthropic/claude-3.5-sonnet", 
        "google/gemini-2.5-flash"
    ]
    
    print(f"🤖 Testing with models: {', '.join(test_models)}")
    print()
    
    # Run benchmark on first image only for quick test
    test_image = images[0]
    print(f"🎯 Running benchmark on: {os.path.basename(test_image)}")
    
    # Extract data with multiple models
    benchmark_results = await benchmark.run_single_benchmark(test_image, test_models)
    
    if not benchmark_results.get("results"):
        print("❌ No benchmark results obtained")
        return
    
    print(f"✅ Extracted data with {len(benchmark_results['results'])} models")
    print()
    
    # Test pairwise comparison
    print("🥊 Testing Pairwise Comparison")
    print("-" * 30)
    
    results = benchmark_results["results"]
    if len(results) >= 2:
        model_a = results[0]
        model_b = results[1]
        
        print(f"Comparing: {model_a['model']} vs {model_b['model']}")
        
        comparison = await benchmark.run_judge_comparison(
            model_a["extraction_result"],
            model_b["extraction_result"],
            test_image,
            model_a["model"],
            model_b["model"]
        )
        
        if comparison["success"]:
            print("✅ Comparison successful!")
            print("\n" + "="*60)
            print(comparison["report"])
            print("="*60)
        else:
            print(f"❌ Comparison failed: {comparison['error']}")
    
    print()
    
    # Test tournament with multiple images if available
    if len(images) > 1:
        print("🏆 Testing Tournament Mode")
        print("-" * 25)
        
        # Run benchmark on first 2 images
        tournament_images = images[:2]
        print(f"Running tournament on {len(tournament_images)} images...")
        
        full_benchmark = await benchmark.run_benchmark(tournament_images, test_models)
        
        if "error" not in full_benchmark:
            tournament_results = await benchmark.run_judge_tournament(full_benchmark)
            
            if "error" not in tournament_results:
                print("✅ Tournament completed!")
                
                # Display rankings
                print("\n🥇 Final Rankings:")
                print("-" * 20)
                
                leaderboard = tournament_results["rankings"]["leaderboard"]
                for entry in leaderboard:
                    print(f"{entry['rank']}. {entry['model']}")
                    print(f"   Win Rate: {entry['win_rate']:.1%}")
                    print(f"   Avg Score: {entry['average_score']:.1f}/10")
                    print(f"   Record: {entry['wins']}W-{entry['losses']}L-{entry['ties']}T")
                    print()
                
                # Display summary
                summary = tournament_results["summary"]
                print(f"📊 Tournament Summary:")
                print(f"   Total Comparisons: {summary['total_comparisons']}")
                print(f"   Successful Judgments: {summary['successful_judgments']}")
                print(f"   Overall Winner: {summary['winner']}")
                
            else:
                print(f"❌ Tournament failed: {tournament_results['error']}")
        else:
            print(f"❌ Benchmark failed: {full_benchmark['error']}")
    
    # Clean up
    await benchmark.judge.close()
    
    print("\n✅ Judge system test completed!")


async def test_caching():
    """Test the caching functionality"""
    print("\n🗄️ Testing Caching System")
    print("-" * 25)
    
    benchmark = StructuredBenchmark()
    
    if not benchmark.judge:
        print("❌ Judge LLM not available")
        return
    
    # Create dummy results for testing
    dummy_result_a = {
        "success": True,
        "data": {
            "dashboard_title": "Test Dashboard A",
            "charts": [{"type": "bar", "values": [1, 2, 3]}],
            "metrics": [{"name": "Revenue", "value": 1000}]
        }
    }
    
    dummy_result_b = {
        "success": True,
        "data": {
            "dashboard_title": "Test Dashboard B", 
            "charts": [{"type": "pie", "values": [4, 5, 6]}],
            "metrics": [{"name": "Revenue", "value": 2000}]
        }
    }
    
    test_image_path = "test_image.jpg"
    
    print("First comparison (should hit API)...")
    start_time = asyncio.get_event_loop().time()
    
    comparison1 = await benchmark.run_judge_comparison(
        dummy_result_a, dummy_result_b, test_image_path, "Model A", "Model B"
    )
    
    first_time = asyncio.get_event_loop().time() - start_time
    
    print("Second comparison (should use cache)...")
    start_time = asyncio.get_event_loop().time()
    
    comparison2 = await benchmark.run_judge_comparison(
        dummy_result_a, dummy_result_b, test_image_path, "Model A", "Model B"
    )
    
    second_time = asyncio.get_event_loop().time() - start_time
    
    print(f"First call time: {first_time:.2f}s")
    print(f"Second call time: {second_time:.2f}s")
    
    if second_time < first_time * 0.1:  # Should be much faster
        print("✅ Caching appears to be working!")
    else:
        print("⚠️ Caching may not be working as expected")
    
    # Check if results are identical
    if (comparison1["success"] and comparison2["success"] and 
        comparison1["judgment"] == comparison2["judgment"]):
        print("✅ Cached results match original!")
    else:
        print("⚠️ Cached results don't match original")
    
    await benchmark.judge.close()


def main():
    """Main function to run all tests"""
    print("🚀 Judge LLM System Test Suite")
    print("=" * 60)
    print()
    
    # Check if we have the required environment
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not found in environment")
        print("Please set your OpenRouter API key to run tests")
        return
    
    # Check if data directory exists
    if not Path("data").exists():
        print("❌ data/ directory not found")
        print("Please ensure test images are available in data/ directory")
        return
    
    # Run tests
    asyncio.run(test_judge_system())
    asyncio.run(test_caching())
    
    print("\n🎉 All tests completed!")


if __name__ == "__main__":
    main()