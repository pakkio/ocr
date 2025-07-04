import httpx
import json
from typing import Optional, Dict, Any, List
from PIL import Image
import asyncio
import os
import glob
from pathlib import Path

from .base import BaseOCRProvider, OCRResult
from ..schemas import DashboardData, QualityAssessment, DASHBOARD_EXTRACTION_PROMPT, QUALITY_ASSESSMENT_PROMPT

class StructuredOCRProvider(BaseOCRProvider):
    """Provider for structured JSON extraction from dashboard images"""
    
    def __init__(self, config):
        super().__init__(config)
        self.provider_name = "structured_openrouter"
        self.api_key = config.openrouter_api_key
        self.base_url = config.openrouter_base_url
        
        if not self.api_key:
            raise ValueError("OpenRouter API key is required for structured extraction")
            
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def discover_data_files(self, data_dir: str = "data") -> List[str]:
        """Auto-discover image files in data directory"""
        supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        image_files = []
        
        data_path = Path(data_dir)
        if not data_path.exists():
            return []
        
        for ext in supported_extensions:
            image_files.extend(glob.glob(str(data_path / ext)))
            image_files.extend(glob.glob(str(data_path / ext.upper())))
        
        # Filter out Zone.Identifier files (Windows download artifacts)
        image_files = [f for f in image_files if not f.endswith('.Identifier')]
        
        return sorted(image_files)
    
    def _normalize_dashboard_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize dashboard data to match Pydantic schema expectations"""
        
        # Normalize charts
        if "charts" in data:
            for chart in data["charts"]:
                # Normalize chart type - convert "Line Chart" to "line", etc.
                if "type" in chart:
                    chart_type = chart["type"]
                    if isinstance(chart_type, str):
                        # Handle different chart type formats
                        normalized_type = chart_type.lower()
                        if normalized_type in ["line chart", "line_chart"]:
                            chart["type"] = "line"
                        elif normalized_type in ["bar chart", "bar_chart"]:
                            chart["type"] = "bar"
                        elif normalized_type in ["pie chart", "pie_chart"]:
                            chart["type"] = "pie"
                        elif normalized_type in ["area chart", "area_chart"]:
                            chart["type"] = "area"
                        elif normalized_type in ["donut chart", "donut_chart"]:
                            chart["type"] = "donut"
                        elif normalized_type in ["gauge chart", "gauge_chart"]:
                            chart["type"] = "gauge"
                        # Keep original if already in correct format
                        elif normalized_type in ["line", "bar", "pie", "area", "donut", "gauge", "table", "metric"]:
                            chart["type"] = normalized_type
                
                # Handle different data field names - some models return 'data' instead of 'data_points'
                if "data" in chart and "data_points" not in chart:
                    chart["data_points"] = chart.pop("data")
                
                # Ensure data_points exists even if empty
                if "data_points" not in chart:
                    chart["data_points"] = []
                
                # Normalize data points
                if "data_points" in chart and isinstance(chart["data_points"], list):
                    for point in chart["data_points"]:
                        # Ensure required fields exist
                        if "label" not in point:
                            point["label"] = point.get("name", point.get("category", "Unknown"))
                        if "value" not in point:
                            point["value"] = point.get("val", point.get("amount", 0))
        
        return data
    
    async def extract_structured_data(
        self, 
        image: Image.Image, 
        model: str = "gpt-4o",
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract structured dashboard data using JSON schema with fallback"""
        
        prompt = custom_prompt or DASHBOARD_EXTRACTION_PROMPT
        
        try:
            # Convert image to base64
            image_b64 = self.image_to_base64(image)
            
            # Check if model supports strict JSON schema
            model_config = self.config.available_models.get(model, {})
            supports_strict_schema = model_config.get("supports_strict_json_schema", True)
            
            # Get Pydantic schema for structured output
            schema = DashboardData.model_json_schema()
            
            # Fix schema for OpenAI strict mode compliance - only for models that support it
            def fix_schema_for_openai(obj):
                if isinstance(obj, dict):
                    # Add required fields if they don't exist and it's an object
                    if 'type' in obj and obj['type'] == 'object' and supports_strict_schema:
                        # Skip additionalProperties to avoid Gradio compatibility issues
                        pass  # obj['additionalProperties'] = False
                        if 'properties' in obj and 'required' not in obj:
                            # For root schema, include ALL fields in required array per OpenAI strict mode
                            if 'dashboard_title' in obj['properties']:
                                obj['required'] = list(obj['properties'].keys())
                            else:
                                # For nested objects, make all non-optional fields required
                                obj['required'] = [k for k, v in obj['properties'].items() 
                                                  if not (isinstance(v, dict) and v.get('default') is not None)]
                    
                    # Recursively fix nested objects
                    for value in obj.values():
                        if isinstance(value, (dict, list)):
                            fix_schema_for_openai(value)
                elif isinstance(obj, list):
                    for item in obj:
                        fix_schema_for_openai(item)
            
            fix_schema_for_openai(schema)
            
            # Ensure schema has title for OpenAI strict mode
            if 'title' not in schema:
                schema['title'] = 'DashboardData'
            
            # Prepare base request
            base_payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "text",
                                "text": prompt + "\n\nIMPORTANT: Respond with valid JSON matching the DashboardData schema."
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
            
            # Add response format based on model capabilities
            if supports_strict_schema:
                base_payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "dashboard_extraction",
                        "strict": True,
                        "schema": schema
                    }
                }
            else:
                base_payload["response_format"] = {
                    "type": "json_object"
                }
            
            payload = base_payload
            
            # Make API request
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
                            content = data['choices'][0]['message']['content']
                            
                            # Parse JSON response, handling markdown code blocks
                            try:
                                # Remove markdown code blocks if present
                                json_content = content.strip()
                                if json_content.startswith('```json'):
                                    json_content = json_content[7:]  # Remove ```json
                                if json_content.startswith('```'):
                                    json_content = json_content[3:]   # Remove ```
                                if json_content.endswith('```'):
                                    json_content = json_content[:-3]  # Remove trailing ```
                                json_content = json_content.strip()
                                
                                structured_data = json.loads(json_content)
                                
                                # Handle case where AI returns a list instead of dict
                                if isinstance(structured_data, list):
                                    if len(structured_data) == 1 and isinstance(structured_data[0], dict):
                                        structured_data = structured_data[0]
                                    else:
                                        return {
                                            "success": False,
                                            "error": f"Expected object but got list with {len(structured_data)} items",
                                            "raw_content": content,
                                            "model": model
                                        }
                                
                                # Ensure we have a dictionary for validation
                                if not isinstance(structured_data, dict):
                                    return {
                                        "success": False,
                                        "error": f"Expected object but got {type(structured_data).__name__}",
                                        "raw_content": content,
                                        "model": model
                                    }
                                
                                # Normalize data before validation
                                normalized_data = self._normalize_dashboard_data(structured_data)
                                
                                # Validate against Pydantic model
                                dashboard_data = DashboardData(**normalized_data)
                                
                                # Return both the validated model and the normalized raw data
                                return {
                                    "success": True,
                                    "data": dashboard_data.model_dump(),
                                    "normalized_data": normalized_data,  # Raw data for quality assessment
                                    "raw_content": content,
                                    "usage": data.get('usage', {}),
                                    "model": model
                                }
                            except json.JSONDecodeError as e:
                                return {
                                    "success": False,
                                    "error": f"JSON parsing error: {str(e)}",
                                    "raw_content": content,
                                    "model": model
                                }
                            except Exception as e:
                                return {
                                    "success": False,
                                    "error": f"Schema validation error: {str(e)}",
                                    "raw_content": content,
                                    "model": model
                                }
                        else:
                            if attempt == self.config.max_retries - 1:
                                return {
                                    "success": False,
                                    "error": f"API Error {response.status_code}: {response.text}",
                                    "model": model
                                }
                            await asyncio.sleep(2 ** attempt)
                            
                    except Exception as e:
                        if attempt == self.config.max_retries - 1:
                            return {
                                "success": False,
                                "error": str(e),
                                "model": model
                            }
                        await asyncio.sleep(2 ** attempt)
                        
        except Exception as e:
            return {
                "success": False,
                "error": f"Provider error: {str(e)}",
                "model": model
            }
    
    async def assess_extraction_quality(
        self,
        extracted_data: Dict[str, Any],
        image_description: str,
        model: str = "gpt-4o-mini"  # Use cheaper model for assessment
    ) -> Dict[str, Any]:
        """Use LLM to assess quality of extracted JSON data"""
        
        # Serialize data properly, handling enums and other non-JSON types
        def json_serializer(obj):
            # Handle enums (like ChartType.LINE)
            if hasattr(obj, 'value') and hasattr(obj, 'name'):
                print(f"DEBUG QUALITY - Serializing enum: {obj} -> {obj.value}")
                return obj.value
            elif hasattr(obj, 'model_dump'):  # Handle Pydantic models
                return obj.model_dump(mode='json')
            else:
                print(f"DEBUG QUALITY - Serializing unknown object: {type(obj)} -> {str(obj)}")
                return str(obj)
        
        # Debug: Check what type of data we're dealing with
        print(f"DEBUG QUALITY - Data type: {type(extracted_data)}")
        print(f"DEBUG QUALITY - Has model_dump: {hasattr(extracted_data, 'model_dump')}")
        print(f"DEBUG QUALITY - Sample data: {str(extracted_data)[:200]}...")
        
        # Pre-process data to convert enums to strings
        def convert_enums_to_strings(obj):
            if isinstance(obj, dict):
                return {k: convert_enums_to_strings(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_enums_to_strings(item) for item in obj]
            elif hasattr(obj, 'value') and hasattr(obj, 'name'):  # Enum
                print(f"DEBUG QUALITY - Converting enum: {obj} -> {obj.value}")
                return obj.value
            else:
                return obj
        
        # Convert enums to strings first
        clean_data = convert_enums_to_strings(extracted_data)
        print(f"DEBUG QUALITY - After enum conversion: {str(clean_data)[:200]}...")
        
        # If the data comes from a Pydantic model, use model_dump(mode='json')
        if hasattr(extracted_data, 'model_dump'):
            serialized_json = json.dumps(extracted_data.model_dump(mode='json'), indent=2)
            print("DEBUG QUALITY - Using model_dump(mode='json')")
        else:
            try:
                serialized_json = json.dumps(clean_data, indent=2)
                print("DEBUG QUALITY - Using json.dumps on clean data")
            except Exception as e:
                print(f"DEBUG QUALITY - JSON serialization error: {e}")
                raise e
        
        prompt = QUALITY_ASSESSMENT_PROMPT.format(
            image_description=image_description,
            extracted_json=serialized_json
        )
        
        try:
            # Check if model supports strict JSON schema
            model_config = self.config.available_models.get(model, {})
            supports_strict_schema = model_config.get("supports_strict_json_schema", True)
            
            # Get assessment schema
            schema = QualityAssessment.model_json_schema()
            
            # Only add additionalProperties: false for models that support strict mode
            if supports_strict_schema:
                def add_additional_properties_false(obj):
                    if isinstance(obj, dict):
                        if 'type' in obj and obj['type'] == 'object':
                            # Skip additionalProperties to avoid Gradio compatibility issues
                            pass  # obj['additionalProperties'] = False
                        for value in obj.values():
                            add_additional_properties_false(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            add_additional_properties_false(item)
                
                add_additional_properties_false(schema)
            
            base_payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt + "\n\nIMPORTANT: Respond with valid JSON matching the QualityAssessment schema."
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.3
            }
            
            # Add response format based on model capabilities
            if supports_strict_schema:
                base_payload["response_format"] = {
                    "type": "json_schema", 
                    "json_schema": {
                        "name": "quality_assessment",
                        "strict": True,
                        "schema": schema
                    }
                }
            else:
                base_payload["response_format"] = {
                    "type": "json_object"
                }
            
            payload = base_payload
            
            print(f"DEBUG QUALITY - API payload model: {payload.get('model')}")
            print(f"DEBUG QUALITY - API payload size: {len(str(payload))} chars")
            print(f"DEBUG QUALITY - Response format: {payload.get('response_format', {}).get('type')}")
            
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                
                print(f"DEBUG QUALITY - API response status: {response.status_code}")
                if response.status_code != 200:
                    try:
                        error_data = response.json()
                        print(f"DEBUG QUALITY - API error response: {error_data}")
                    except:
                        print(f"DEBUG QUALITY - API error text: {response.text}")
                
                if response.status_code == 200:
                    data = response.json()
                    content = data['choices'][0]['message']['content']
                    
                    try:
                        # Remove markdown code blocks if present
                        json_content = content.strip()
                        if json_content.startswith('```json'):
                            json_content = json_content[7:]  # Remove ```json
                        if json_content.startswith('```'):
                            json_content = json_content[3:]   # Remove ```
                        if json_content.endswith('```'):
                            json_content = json_content[:-3]  # Remove trailing ```
                        json_content = json_content.strip()
                        
                        assessment_data = json.loads(json_content)
                        
                        # Handle nested responses (some models wrap the response)
                        if len(assessment_data) == 1 and any(key.lower() in ['quality_assessment', 'assessment', 'qualityassessment'] for key in assessment_data.keys()):
                            assessment_data = list(assessment_data.values())[0]
                        
                        # Normalize field names - different models use different conventions
                        field_mappings = {
                            'completeness': 'completeness_score',
                            'accuracy': 'accuracy_score', 
                            'structure': 'structure_score',
                            'confidence': 'confidence_level'
                        }
                        
                        for old_key, new_key in field_mappings.items():
                            if old_key in assessment_data and new_key not in assessment_data:
                                assessment_data[new_key] = assessment_data[old_key]
                                del assessment_data[old_key]
                        
                        # Normalize recommendations field - convert string to list if needed
                        if 'recommendations' in assessment_data and isinstance(assessment_data['recommendations'], str):
                            assessment_data['recommendations'] = [assessment_data['recommendations']]
                        
                        # Ensure all required fields exist with defaults
                        assessment_data.setdefault('missing_elements', [])
                        assessment_data.setdefault('potential_errors', [])
                        assessment_data.setdefault('recommendations', [])
                        
                        assessment = QualityAssessment(**assessment_data)
                        return {
                            "success": True,
                            "assessment": assessment.model_dump(),
                            "usage": data.get('usage', {})
                        }
                    except Exception as e:
                        return {
                            "success": False,
                            "error": f"Assessment parsing error: {str(e)}",
                            "raw_content": content
                        }
                else:
                    return {
                        "success": False,
                        "error": f"Assessment API error: {response.status_code}"
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "error": f"Assessment error: {str(e)}"
            }
    
    async def extract_text(
        self, 
        image: Image.Image, 
        model: str = "gpt-4o",
        prompt: Optional[str] = None
    ) -> OCRResult:
        """Extract text maintaining compatibility with base interface"""
        
        structured_result = await self.extract_structured_data(image, model, prompt)
        
        if structured_result["success"]:
            # Convert structured data to text format for compatibility
            data = structured_result["data"]
            text_parts = []
            
            # Add dashboard title
            if data.get("dashboard_title"):
                text_parts.append(f"Dashboard: {data['dashboard_title']}")
            
            # Add metrics
            for metric in data.get("metrics", []):
                text_parts.append(f"{metric['label']}: {metric['value']}{metric.get('units', '')}")
            
            # Add chart data
            for chart in data.get("charts", []):
                if chart.get("title"):
                    text_parts.append(f"Chart: {chart['title']}")
                for point in chart.get("data_points", []):
                    text_parts.append(f"  {point['label']}: {point['value']}")
            
            # Add other text content
            text_parts.extend(data.get("text_content", []))
            
            combined_text = "\n".join(text_parts)
            
            return OCRResult(
                text=combined_text,
                execution_time=0,  # Will be set by decorator
                confidence=0.95,  # High confidence for structured extraction
                provider=self.provider_name,
                model=model,
                metadata={
                    "structured_data": data,
                    "extraction_method": "json_schema",
                    "usage": structured_result.get("usage", {})
                }
            )
        else:
            return OCRResult(
                text="",
                execution_time=0,
                error=structured_result["error"],
                provider=self.provider_name,
                model=model
            )
    
    async def run_comprehensive_benchmark(
        self,
        data_dir: str = "data",
        models: List[str] = None
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark on all images in data directory"""
        
        if models is None:
            models = ["gpt-4o", "anthropic/claude-3.5-sonnet", "gpt-4-vision-preview"]
        
        # Discover images
        image_files = self.discover_data_files(data_dir)
        if not image_files:
            return {"error": "No images found in data directory"}
        
        results = {
            "summary": {
                "total_images": len(image_files),
                "models_tested": models,
                "timestamp": asyncio.get_event_loop().time()
            },
            "results": [],
            "quality_assessments": []
        }
        
        for image_path in image_files:
            image_name = os.path.basename(image_path)
            
            try:
                image = Image.open(image_path)
                
                for model in models:
                    # Extract structured data
                    extraction_result = await self.extract_structured_data(image, model)
                    
                    result = {
                        "image_path": image_path,
                        "image_name": image_name,
                        "model": model,
                        "extraction_result": extraction_result
                    }
                    
                    # If extraction succeeded, assess quality
                    if extraction_result["success"]:
                        image_description = f"Dashboard image: {image_name}"
                        quality_result = await self.assess_extraction_quality(
                            extraction_result["data"],
                            image_description,
                            "gpt-4o-mini"  # Cheaper model for assessment
                        )
                        result["quality_assessment"] = quality_result
                    
                    results["results"].append(result)
                    
            except Exception as e:
                results["results"].append({
                    "image_path": image_path,
                    "image_name": image_name,
                    "error": str(e)
                })
        
        return results