"""
Judge LLM system for comparing OCR extraction results using Google Gemini Flash 2.0.
Implements caching to avoid redundant API calls and provides human-readable comparisons.
"""

import json
import hashlib
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import asyncio
import httpx
from pydantic import BaseModel, Field

from .config import config


class JudgmentResult(BaseModel):
    """Structured judgment result from the judge LLM"""
    winner: str = Field(description="'result_a', 'result_b', or 'tie'")
    confidence: float = Field(description="Confidence score 0.0-1.0", ge=0.0, le=1.0)
    reasoning: str = Field(description="Human-readable explanation of the decision")
    criteria_scores: Dict[str, Dict[str, float]] = Field(
        description="Detailed scores for each result on different criteria"
    )
    overall_scores: Dict[str, float] = Field(
        description="Overall scores for result_a and result_b"
    )


class JudgeCache:
    """Simple file-based cache for judge results to avoid redundant API calls"""
    
    def __init__(self, cache_dir: str = "cache/judge_results"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _generate_cache_key(self, result_a: Dict[str, Any], result_b: Dict[str, Any], 
                          image_path: str, criteria: List[str]) -> str:
        """Generate a unique cache key for the comparison"""
        combined_data = {
            "result_a": result_a,
            "result_b": result_b,
            "image_path": image_path,
            "criteria": sorted(criteria)
        }
        data_str = json.dumps(combined_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def get(self, cache_key: str, max_age_hours: int = 24) -> Optional[JudgmentResult]:
        """Retrieve cached judgment if it exists and is not too old"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            # Check if cache is too old
            cached_time = datetime.fromisoformat(cached_data["timestamp"])
            if datetime.now() - cached_time > timedelta(hours=max_age_hours):
                return None
                
            return JudgmentResult(**cached_data["judgment"])
        except Exception:
            return None
    
    def set(self, cache_key: str, judgment: JudgmentResult):
        """Store judgment result in cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "judgment": judgment.model_dump()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)


class JudgeLLM:
    """Judge LLM system using Google Gemini Flash 2.0 for OCR result comparison"""
    
    def __init__(self, judge_model: str = "google/gemini-2.5-flash"):
        self.judge_model = judge_model
        self.cache = JudgeCache()
        self.client = httpx.AsyncClient(
            base_url=config.openrouter_base_url,
            headers={
                "Authorization": f"Bearer {config.openrouter_api_key}",
                "Content-Type": "application/json"
            },
            timeout=config.timeout_seconds
        )
    
    def _create_judgment_prompt(self, result_a: Dict[str, Any], result_b: Dict[str, Any], 
                               image_description: str, criteria: List[str]) -> str:
        """Create the prompt for the judge LLM"""
        
        criteria_desc = "\n".join([f"- {criterion}" for criterion in criteria])
        
        prompt = f"""
You are an expert judge evaluating two OCR extraction results from the same dashboard/analytics image.

**Image Context:** {image_description}

**Your Task:** Compare the two OCR extraction results and determine which one is better overall.

**Evaluation Criteria:**
{criteria_desc}

**Result A:**
```json
{json.dumps(result_a, indent=2)}
```

**Result B:**
```json
{json.dumps(result_b, indent=2)}
```

**Instructions:**
1. Evaluate both results against each criterion (score 0-10 for each)
2. Provide an overall winner: "result_a", "result_b", or "tie"
3. Give a confidence score (0.0-1.0) for your decision
4. Provide clear, human-readable reasoning for your judgment

**Important:** Focus on accuracy, completeness, structure, and practical utility. Consider:
- Which result captures more relevant data from the dashboard?
- Which has better structured organization?
- Which would be more useful for downstream analysis?
- Which handles complex layouts and overlapping elements better?

Respond with a structured analysis in the following JSON format:
{{
    "winner": "result_a|result_b|tie",
    "confidence": 0.85,
    "reasoning": "Clear explanation of why this result is better...",
    "criteria_scores": {{
        "result_a": {{
            "accuracy": 8.5,
            "completeness": 7.0,
            "structure": 9.0,
            "utility": 8.0
        }},
        "result_b": {{
            "accuracy": 7.0,
            "completeness": 8.5,
            "structure": 6.5,
            "utility": 7.5
        }}
    }},
    "overall_scores": {{
        "result_a": 8.1,
        "result_b": 7.4
    }}
}}
"""
        return prompt
    
    async def judge_comparison(self, 
                             result_a: Dict[str, Any], 
                             result_b: Dict[str, Any],
                             image_path: str,
                             image_description: str = "",
                             criteria: List[str] = None,
                             use_cache: bool = True) -> JudgmentResult:
        """
        Compare two OCR extraction results using the judge LLM
        
        Args:
            result_a: First OCR extraction result
            result_b: Second OCR extraction result  
            image_path: Path to the source image
            image_description: Optional description of the image content
            criteria: List of evaluation criteria
            use_cache: Whether to use caching
            
        Returns:
            JudgmentResult with winner, confidence, and detailed reasoning
        """
        
        if criteria is None:
            criteria = [
                "Accuracy: How well does the result match the actual content?",
                "Completeness: How much of the relevant data was captured?", 
                "Structure: How well-organized and usable is the extracted data?",
                "Utility: How useful would this be for further analysis?"
            ]
        
        # Check cache first
        if use_cache:
            cache_key = self.cache._generate_cache_key(result_a, result_b, image_path, criteria)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result
        
        # Generate image description if not provided
        if not image_description:
            image_description = f"Dashboard/analytics image from {os.path.basename(image_path)}"
        
        # Create judgment prompt
        prompt = self._create_judgment_prompt(result_a, result_b, image_description, criteria)
        
        # Make API call to judge model
        try:
            response = await self.client.post(
                "/chat/completions",
                json={
                    "model": self.judge_model,
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are an expert OCR evaluation judge. Provide structured, objective comparisons."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.1,  # Low temperature for consistent judgments
                    "max_tokens": 1500
                }
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Parse the JSON response
            judgment_data = json.loads(content)
            judgment = JudgmentResult(**judgment_data)
            
            # Cache the result
            if use_cache:
                self.cache.set(cache_key, judgment)
            
            return judgment
            
        except Exception as e:
            # Fallback judgment in case of API failure
            return JudgmentResult(
                winner="tie",
                confidence=0.0,
                reasoning=f"Unable to perform judgment due to error: {str(e)}",
                criteria_scores={
                    "result_a": {criterion.split(":")[0].lower(): 5.0 for criterion in criteria},
                    "result_b": {criterion.split(":")[0].lower(): 5.0 for criterion in criteria}
                },
                overall_scores={"result_a": 5.0, "result_b": 5.0}
            )
    
    async def batch_judge_comparisons(self, 
                                    comparisons: List[Tuple[Dict[str, Any], Dict[str, Any], str]],
                                    criteria: List[str] = None) -> List[JudgmentResult]:
        """
        Perform multiple judgments in batch with proper rate limiting
        
        Args:
            comparisons: List of (result_a, result_b, image_path) tuples
            criteria: Evaluation criteria to use for all comparisons
            
        Returns:
            List of JudgmentResult objects
        """
        
        tasks = []
        for result_a, result_b, image_path in comparisons:
            task = self.judge_comparison(result_a, result_b, image_path, criteria=criteria)
            tasks.append(task)
        
        # Execute with some delay to respect rate limits
        results = []
        for task in tasks:
            result = await task
            results.append(result)
            await asyncio.sleep(0.5)  # Small delay between requests
            
        return results
    
    def create_human_readable_report(self, judgment: JudgmentResult, 
                                   model_a_name: str, model_b_name: str) -> str:
        """Create a human-readable report from the judgment result"""
        
        winner_name = model_a_name if judgment.winner == "result_a" else (
            model_b_name if judgment.winner == "result_b" else "Tie"
        )
        
        report = f"""
## OCR Comparison Judgment

**Winner:** {winner_name} (Confidence: {judgment.confidence:.1%})

**Reasoning:**
{judgment.reasoning}

**Detailed Scores:**

| Criterion | {model_a_name} | {model_b_name} |
|-----------|----------------|----------------|
"""
        
        for criterion in judgment.criteria_scores.get("result_a", {}).keys():
            score_a = judgment.criteria_scores["result_a"].get(criterion, 0)
            score_b = judgment.criteria_scores["result_b"].get(criterion, 0)
            report += f"| {criterion.title()} | {score_a:.1f}/10 | {score_b:.1f}/10 |\n"
        
        report += f"""
**Overall Scores:**
- {model_a_name}: {judgment.overall_scores.get("result_a", 0):.1f}/10
- {model_b_name}: {judgment.overall_scores.get("result_b", 0):.1f}/10
"""
        
        return report
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


# Convenience function for quick comparisons
async def judge_ocr_results(result_a: Dict[str, Any], 
                          result_b: Dict[str, Any],
                          image_path: str,
                          model_a_name: str = "Model A",
                          model_b_name: str = "Model B") -> str:
    """
    Quick function to judge two OCR results and return a human-readable report
    
    Args:
        result_a: First OCR extraction result
        result_b: Second OCR extraction result
        image_path: Path to source image
        model_a_name: Name of first model for reporting
        model_b_name: Name of second model for reporting
        
    Returns:
        Human-readable comparison report
    """
    
    judge = JudgeLLM()
    try:
        judgment = await judge.judge_comparison(result_a, result_b, image_path)
        report = judge.create_human_readable_report(judgment, model_a_name, model_b_name)
        return report
    finally:
        await judge.close()