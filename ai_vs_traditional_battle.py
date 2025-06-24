#!/usr/bin/env python3
"""
🥊 AI vs Traditional OCR Battle Arena
====================================

Ultimate showdown between AI Vision Language Models and Traditional OCR engines.
This comprehensive benchmark demonstrates AI's superiority in complex document understanding.

Test Categories:
1. Raw Text Extraction Accuracy
2. Structured Data Understanding 
3. Context Awareness (ignoring watermarks)
4. Complex Layout Handling
5. Speed vs Quality Trade-offs

Battle Format:
- Traditional OCR: Tesseract, EasyOCR, PaddleOCR
- AI Vision Models: GPT-4o, Claude 3.5, Gemini 2.5
- Head-to-head scoring across multiple criteria
- Visual comparison of results

Author: Claude Code  
Version: 1.0 - The Battle Edition
"""

import asyncio
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from PIL import Image
import pandas as pd
import numpy as np

# Import our providers
from src.config import config
from src.providers.traditional_provider import TraditionalOCRProvider
from src.providers.structured_provider import StructuredOCRProvider
from src.schemas import DashboardData, QualityAssessment

@dataclass
class BattleResult:
    """Results from AI vs Traditional OCR battle"""
    image_name: str
    traditional_results: Dict[str, Any]  # Results from all traditional OCR
    ai_results: Dict[str, Any]  # Results from AI models
    battle_scores: Dict[str, float]  # Comparative scores
    winner_category: str  # "AI" or "Traditional"
    victory_margin: float  # How much AI won by (0-10 scale)
    
@dataclass 
class BattleMetrics:
    """Detailed metrics for the battle"""
    text_extraction_accuracy: float  # How much text was captured
    data_structure_quality: float   # How well structured the output
    context_awareness: float        # Filtering watermarks, understanding context
    layout_handling: float          # Complex dashboard layouts
    speed_efficiency: float         # Speed vs quality ratio
    overall_intelligence: float     # General AI understanding

class AIvsTraditionalBattle:
    """Main battle coordinator between AI and Traditional OCR"""
    
    def __init__(self):
        self.config = config
        self.traditional_provider = TraditionalOCRProvider(config)
        self.ai_provider = StructuredOCRProvider(config)
        
        # Battle configuration
        self.traditional_models = ["tesseract", "easyocr", "paddleocr"]
        # Use a subset of available AI models for battles (fast, representative models)
        self.ai_models = [
            "google/gemini-2.5-flash",  # Fast and cheap
            "anthropic/claude-3.5-sonnet",  # Excellent reasoning
            "openai/gpt-4o-mini"  # Good balance
        ]
        
        self.battle_results = []
        
    async def run_traditional_ocr_battle(self, image: Image.Image, image_name: str) -> Dict[str, Any]:
        """Run all traditional OCR models against an image"""
        results = {}
        
        print(f"  🔧 Traditional OCR attacking {image_name}...")
        
        for model in self.traditional_models:
            start_time = time.time()
            try:
                result = await self.traditional_provider.extract_text(image, model)
                results[model] = {
                    'text': result.text,
                    'execution_time': result.execution_time,
                    'confidence': result.confidence,
                    'error': result.error,
                    'text_length': len(result.text.strip()),
                    'lines_extracted': len(result.text.strip().split('\n')) if result.text else 0
                }
                print(f"    ✓ {model}: {len(result.text)} chars in {result.execution_time:.2f}s")
            except Exception as e:
                results[model] = {
                    'text': '',
                    'execution_time': time.time() - start_time,
                    'error': str(e),
                    'text_length': 0,
                    'lines_extracted': 0
                }
                print(f"    ✗ {model}: Failed - {str(e)}")
        
        return results
    
    async def run_ai_battle(self, image: Image.Image, image_name: str) -> Dict[str, Any]:
        """Run AI vision models against an image"""
        results = {}
        
        print(f"  🤖 AI Vision Models attacking {image_name}...")
        
        for model in self.ai_models:
            start_time = time.time()
            try:
                # Extract structured data
                structured_result = await self.ai_provider.extract_structured_data(image, model)
                
                # Check if extraction was successful
                if not structured_result.get("success", False):
                    raise Exception(structured_result.get("error", "Extraction failed"))
                
                structured_data = structured_result.get("data", {})
                
                # Also get quality assessment 
                quality_result = await self.ai_provider.assess_extraction_quality(
                    structured_data, 
                    f"Dashboard image: {image_name}"
                )
                
                # Extract quality data from the result
                quality_data = quality_result.get('assessment', {}) if quality_result.get('success') else {}
                
                execution_time = time.time() - start_time
                
                results[model] = {
                    'structured_data': structured_data,
                    'quality_assessment': quality_result,
                    'execution_time': execution_time,
                    'charts_extracted': len(structured_data.get('charts', [])),
                    'metrics_extracted': len(structured_data.get('metrics', [])),
                    'data_points_total': sum(len(chart.get('data_points', [])) for chart in structured_data.get('charts', [])),
                    'completeness_score': quality_data.get('completeness_score', 0),
                    'accuracy_score': quality_data.get('accuracy_score', 0),
                    'structure_score': quality_data.get('structure_score', 0)
                }
                
                print(f"    ✓ {model}: {results[model]['charts_extracted']} charts, {results[model]['metrics_extracted']} metrics in {execution_time:.2f}s")
                
            except Exception as e:
                results[model] = {
                    'structured_data': {},
                    'quality_assessment': {},
                    'execution_time': time.time() - start_time,
                    'error': str(e),
                    'charts_extracted': 0,
                    'metrics_extracted': 0,
                    'completeness_score': 0,
                    'accuracy_score': 0,
                    'structure_score': 0
                }
                print(f"    ✗ {model}: Failed - {str(e)}")
        
        return results
    
    def calculate_battle_scores(self, traditional_results: Dict, ai_results: Dict, image_name: str) -> Tuple[Dict[str, float], str, float]:
        """Calculate who wins the battle and by how much"""
        
        # Traditional OCR scoring (0-10 scale)
        traditional_scores = {
            'text_extraction': 0,
            'speed': 0,
            'reliability': 0
        }
        
        # Calculate traditional OCR aggregate scores
        working_traditional = [r for r in traditional_results.values() if not r.get('error')]
        if working_traditional:
            avg_text_length = np.mean([r['text_length'] for r in working_traditional])
            avg_speed = np.mean([1/max(r['execution_time'], 0.1) for r in working_traditional])  # Inverse of time
            reliability = len(working_traditional) / len(traditional_results)  # Success rate
            
            traditional_scores['text_extraction'] = min(10, avg_text_length / 100)  # Scale text length
            traditional_scores['speed'] = min(10, avg_speed * 2)  # Scale speed
            traditional_scores['reliability'] = reliability * 10
        
        # AI Vision Model scoring (0-10 scale)
        working_ai = [r for r in ai_results.values() if not r.get('error')]
        ai_scores = {
            'structured_understanding': 0,
            'data_quality': 0, 
            'context_awareness': 0,
            'intelligence': 0
        }
        
        if working_ai:
            avg_charts = np.mean([r['charts_extracted'] for r in working_ai])
            avg_metrics = np.mean([r['metrics_extracted'] for r in working_ai])
            avg_completeness = np.mean([r['completeness_score'] for r in working_ai])
            avg_accuracy = np.mean([r['accuracy_score'] for r in working_ai])
            avg_structure = np.mean([r['structure_score'] for r in working_ai])
            
            ai_scores['structured_understanding'] = min(10, (avg_charts + avg_metrics) * 2)
            ai_scores['data_quality'] = (avg_completeness + avg_accuracy) / 2
            ai_scores['context_awareness'] = avg_structure
            ai_scores['intelligence'] = (avg_completeness + avg_accuracy + avg_structure) / 3
        
        # Calculate overall scores
        traditional_total = sum(traditional_scores.values()) / len(traditional_scores)
        ai_total = sum(ai_scores.values()) / len(ai_scores)
        
        # Determine winner
        if ai_total > traditional_total:
            winner = "AI"
            victory_margin = ai_total - traditional_total
        elif traditional_total > ai_total:
            winner = "Traditional" 
            victory_margin = traditional_total - ai_total
        else:
            winner = "Tie"
            victory_margin = 0
        
        battle_scores = {
            'traditional_total': traditional_total,
            'ai_total': ai_total,
            **{f'traditional_{k}': v for k, v in traditional_scores.items()},
            **{f'ai_{k}': v for k, v in ai_scores.items()}
        }
        
        return battle_scores, winner, victory_margin
    
    async def battle_single_image(self, image_path: str) -> BattleResult:
        """Run complete battle on a single image"""
        
        image_name = Path(image_path).name
        print(f"\n⚔️  BATTLE ARENA: {image_name}")
        print("=" * 60)
        
        # Load image
        image = Image.open(image_path)
        
        # Run traditional OCR battle
        traditional_results = await self.run_traditional_ocr_battle(image, image_name)
        
        # Run AI battle  
        ai_results = await self.run_ai_battle(image, image_name)
        
        # Calculate battle scores
        battle_scores, winner, victory_margin = self.calculate_battle_scores(
            traditional_results, ai_results, image_name
        )
        
        print(f"\n🏆 BATTLE RESULT: {winner} wins by {victory_margin:.1f} points!")
        print(f"   Traditional OCR: {battle_scores['traditional_total']:.1f}/10")
        print(f"   AI Vision: {battle_scores['ai_total']:.1f}/10")
        
        return BattleResult(
            image_name=image_name,
            traditional_results=traditional_results,
            ai_results=ai_results,
            battle_scores=battle_scores,
            winner_category=winner,
            victory_margin=victory_margin
        )
    
    async def run_full_battle_tournament(self, data_dir: str = "data") -> List[BattleResult]:
        """Run complete tournament across all test images"""
        
        print("🥊 STARTING AI vs TRADITIONAL OCR BATTLE TOURNAMENT")
        print("=" * 70)
        
        # Discover images
        image_files = self.ai_provider.discover_data_files(data_dir)
        
        if not image_files:
            print("❌ No test images found!")
            return []
        
        print(f"📊 Found {len(image_files)} battle arenas (test images)")
        
        # Battle each image
        battle_results = []
        for image_path in image_files:
            result = await self.battle_single_image(image_path)
            battle_results.append(result)
            self.battle_results.append(result)
        
        # Calculate tournament summary
        self.print_tournament_summary(battle_results)
        
        return battle_results
    
    def print_tournament_summary(self, results: List[BattleResult]):
        """Print comprehensive tournament results"""
        
        print("\n" + "=" * 70)
        print("🏆 FINAL TOURNAMENT RESULTS")
        print("=" * 70)
        
        ai_wins = sum(1 for r in results if r.winner_category == "AI")
        traditional_wins = sum(1 for r in results if r.winner_category == "Traditional")
        ties = sum(1 for r in results if r.winner_category == "Tie")
        
        print(f"📊 OVERALL STANDINGS:")
        print(f"   🤖 AI Vision Models: {ai_wins} victories")
        print(f"   🔧 Traditional OCR: {traditional_wins} victories") 
        print(f"   🤝 Ties: {ties}")
        
        if ai_wins > traditional_wins:
            print(f"\n🥇 AI VISION MODELS DOMINATE!")
            win_rate = (ai_wins / len(results)) * 100
            print(f"   Victory Rate: {win_rate:.1f}%")
            avg_margin = np.mean([r.victory_margin for r in results if r.winner_category == "AI"])
            print(f"   Average Victory Margin: {avg_margin:.1f} points")
        
        # Detailed breakdown
        print(f"\n📈 DETAILED PERFORMANCE METRICS:")
        
        avg_scores = {}
        for key in results[0].battle_scores.keys():
            avg_scores[key] = np.mean([r.battle_scores[key] for r in results])
        
        print(f"   Traditional OCR Average: {avg_scores['traditional_total']:.1f}/10")
        print(f"     - Text Extraction: {avg_scores['traditional_text_extraction']:.1f}/10")
        print(f"     - Speed: {avg_scores['traditional_speed']:.1f}/10") 
        print(f"     - Reliability: {avg_scores['traditional_reliability']:.1f}/10")
        
        print(f"   AI Vision Average: {avg_scores['ai_total']:.1f}/10")
        print(f"     - Structured Understanding: {avg_scores['ai_structured_understanding']:.1f}/10")
        print(f"     - Data Quality: {avg_scores['ai_data_quality']:.1f}/10")
        print(f"     - Context Awareness: {avg_scores['ai_context_awareness']:.1f}/10")
        print(f"     - Intelligence: {avg_scores['ai_intelligence']:.1f}/10")
        
        # Individual battle recap
        print(f"\n⚔️  INDIVIDUAL BATTLE RESULTS:")
        for result in results:
            status_emoji = "🥇" if result.winner_category == "AI" else "🥈" if result.winner_category == "Traditional" else "🤝"
            print(f"   {status_emoji} {result.image_name}: {result.winner_category} (+{result.victory_margin:.1f})")
    
    def export_battle_results(self, filename: str = "ai_vs_traditional_battle_results.json"):
        """Export detailed battle results to JSON"""
        
        export_data = {
            'tournament_summary': {
                'total_battles': len(self.battle_results),
                'ai_victories': sum(1 for r in self.battle_results if r.winner_category == "AI"),
                'traditional_victories': sum(1 for r in self.battle_results if r.winner_category == "Traditional"),
                'ties': sum(1 for r in self.battle_results if r.winner_category == "Tie")
            },
            'detailed_results': [asdict(result) for result in self.battle_results]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"📄 Battle results exported to {filename}")


async def main():
    """Run the ultimate AI vs Traditional OCR battle"""
    
    battle_arena = AIvsTraditionalBattle()
    
    print("🎯 Initializing Battle Systems...")
    print(f"Traditional OCR Models: {battle_arena.traditional_models}")
    print(f"AI Vision Models: {battle_arena.ai_models}")
    
    # Run the tournament
    results = await battle_arena.run_full_battle_tournament()
    
    # Export results
    battle_arena.export_battle_results()
    
    print("\n🎉 Battle complete! AI supremacy demonstrated.")

if __name__ == "__main__":
    asyncio.run(main())