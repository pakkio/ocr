#!/usr/bin/env python3
"""
🚀 Quick Test Runner
===================

Simple script to run the comprehensive test suite with different options.
"""

import asyncio
import argparse
import sys
from comprehensive_test_suite import ComprehensiveTestSuite

async def run_quick_test():
    """Run a quick test with a subset of models"""
    print("🏃‍♂️ Running Quick Test (subset of models)...")
    
    suite = ComprehensiveTestSuite()
    
    # Override with smaller model set for quick testing
    suite.vlm_models = {
        "OpenAI": ["gpt-4o", "openai/gpt-4o-mini"],
        "Anthropic": ["anthropic/claude-3.5-sonnet"], 
        "Google": ["google/gemini-2.5-flash"]
    }
    suite.traditional_models = ["easyocr"]  # Just one traditional model
    suite.test_images = suite.test_images[:1]  # Just one image
    
    summary = await suite.run_all_tests()
    return summary

async def run_full_test():
    """Run the complete test suite"""
    print("🏗️ Running Full Test Suite...")
    
    suite = ComprehensiveTestSuite()
    summary = await suite.run_all_tests()
    return summary

async def run_provider_test(provider: str):
    """Run tests for a specific provider only"""
    print(f"🎯 Running {provider} Provider Test...")
    
    suite = ComprehensiveTestSuite()
    
    if provider.lower() == "openai":
        suite.vlm_models = {"OpenAI": suite.vlm_models["OpenAI"]}
    elif provider.lower() == "anthropic":
        suite.vlm_models = {"Anthropic": suite.vlm_models["Anthropic"]}
    elif provider.lower() == "google":
        suite.vlm_models = {"Google": suite.vlm_models["Google"]}
    elif provider.lower() == "traditional":
        suite.vlm_models = {}  # No VLM models
    else:
        print(f"❌ Unknown provider: {provider}")
        return None
    
    summary = await suite.run_all_tests()
    return summary

def main():
    parser = argparse.ArgumentParser(description="OCR Test Suite Runner")
    parser.add_argument("--mode", choices=["quick", "full", "provider"], default="quick",
                       help="Test mode to run")
    parser.add_argument("--provider", choices=["openai", "anthropic", "google", "traditional"],
                       help="Specific provider to test (use with --mode provider)")
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        summary = asyncio.run(run_quick_test())
    elif args.mode == "full":
        summary = asyncio.run(run_full_test())
    elif args.mode == "provider":
        if not args.provider:
            print("❌ Provider must be specified with --mode provider")
            sys.exit(1)
        summary = asyncio.run(run_provider_test(args.provider))
    
    if summary:
        print(f"\n✅ Test completed: {summary.success_rate:.1f}% success rate")
    else:
        print("\n❌ Test failed to complete")

if __name__ == "__main__":
    main()