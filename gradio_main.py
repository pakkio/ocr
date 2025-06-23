"""
Modern Gradio interface for the Advanced OCR Benchmark Suite.
FINAL, WORKING FIX: This version resolves the persistent `TypeError` on startup.

Root Cause: A bug in Gradio's schema generation for the `gr.Examples` component.
The component was causing the introspection engine to fail during app launch,
leading to the `TypeError: argument of type 'bool' is not iterable`.

Solution: The `gr.Examples` components have been removed entirely. This prevents
the buggy schema generation from ever running and allows the application to launch
correctly. The core functionality remains unchanged.
"""
import gradio as gr
import asyncio
import json
import os
import httpx
import base64
import io
import time
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from PIL import Image

# --- Dependency Availability Checks ---
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# --- Standalone Logic Classes (Pydantic-Free) ---

DASHBOARD_SCHEMA_DICT = {
    "dashboard_title": "string (optional)",
    "charts": [{"title": "string (optional)", "type": "string (e.g., 'pie', 'bar')", "data_points": [{"label": "string", "value": "number or string"}]}],
    "metrics": [{"label": "string", "value": "string or number", "units": "string (optional)"}],
    "time_series": [{"title": "string (optional)", "data": [{"period": "string", "value": "number"}]}],
    "text_content": ["string"]
}

class StandaloneStructuredOCR:
    """Performs structured OCR via OpenRouter without any src/ Pydantic dependencies."""
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        self.schema_prompt = json.dumps(DASHBOARD_SCHEMA_DICT, indent=2)

    @staticmethod
    def image_to_base64(image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    async def _make_api_call(self, payload: Dict) -> Dict:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(f"{self.base_url}/chat/completions", headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()

    async def extract(self, image: Image.Image, model: str) -> Dict[str, Any]:
        if not self.api_key: return {"success": False, "error": "OPENROUTER_API_KEY not set."}
        prompt = (
            "Analyze the dashboard image and extract all visible data. Your output MUST be a single, valid JSON object "
            f"that conforms to this structure: \n```json\n{self.schema_prompt}\n```"
        )
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.image_to_base64(image)}"}}] }],
            "response_format": {"type": "json_object"}
        }
        try:
            response_data = await self._make_api_call(payload)
            content = response_data['choices'][0]['message']['content']
            
            # Remove markdown code blocks if present
            json_content = content.strip()
            if json_content.startswith('```json'):
                json_content = json_content[7:]  # Remove ```json
            if json_content.startswith('```'):
                json_content = json_content[3:]   # Remove ```
            if json_content.endswith('```'):
                json_content = json_content[:-3]  # Remove trailing ```
            json_content = json_content.strip()
            
            return {"success": True, "data": json.loads(json_content)}
        except Exception as e:
            return {"success": False, "error": f"API call for {model} failed: {str(e)}"}

    async def assess(self, extracted_data: Dict, model: str = "anthropic/claude-3.5-haiku") -> Dict[str, Any]:
        prompt = (
            "You are a data quality analyst. Evaluate the provided JSON data from a dashboard. "
            "Your output MUST be a single, valid JSON object with keys: 'completeness_score', 'accuracy_score', 'structure_score', 'confidence_level', 'recommendations'.\n\n"
            f"EXTRACTED JSON:\n```json\n{json.dumps(extracted_data, indent=2)}\n```"
        )
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}}
        try:
            response_data = await self._make_api_call(payload)
            content = response_data['choices'][0]['message']['content']
            
            # Remove markdown code blocks if present
            json_content = content.strip()
            if json_content.startswith('```json'):
                json_content = json_content[7:]  # Remove ```json
            if json_content.startswith('```'):
                json_content = json_content[3:]   # Remove ```
            if json_content.endswith('```'):
                json_content = json_content[:-3]  # Remove trailing ```
            json_content = json_content.strip()
            
            return {"success": True, "assessment": json.loads(json_content)}
        except Exception as e:
            return {"success": False, "error": f"Assessment failed: {str(e)}"}

class StandaloneTraditionalOCR:
    """Performs traditional OCR without any src/ Pydantic dependencies."""
    def __init__(self):
        self.easyocr_reader = easyocr.Reader(['en']) if EASYOCR_AVAILABLE else None
        self.paddle_reader = PaddleOCR(use_angle_cls=True, lang='en', show_log=False) if PADDLEOCR_AVAILABLE else None

    async def extract_text(self, image: Image.Image, model: str) -> Dict:
        start_time = time.time()
        try:
            img_array = np.array(image.convert('RGB'))
            if model == "easyocr":
                if not self.easyocr_reader: raise ImportError("EasyOCR not available.")
                results = self.easyocr_reader.readtext(img_array)
                text = " ".join([res[1] for res in results])
                conf = np.mean([res[2] for res in results]) if results else 0.0
            elif model == "paddleocr":
                if not self.paddle_reader: raise ImportError("PaddleOCR not available.")
                results = self.paddle_reader.ocr(img_array, cls=True)
                lines = results[0] if results and results[0] else []
                text = " ".join([line[1][0] for line in lines])
                conf = np.mean([line[1][1] for line in lines]) if lines else 0.0
            elif model == "tesseract":
                if not TESSERACT_AVAILABLE: raise ImportError("Tesseract not available.")
                text = pytesseract.image_to_string(image)
                conf = 0.0
            else:
                raise ValueError(f"Unknown traditional model: {model}")
            return {"model": model, "success": True, "text": text.strip(), "confidence": conf, "execution_time": time.time() - start_time, "error": None}
        except Exception as e:
            return {"model": model, "success": False, "text": None, "confidence": 0.0, "execution_time": time.time() - start_time, "error": str(e)}

# --- UI Formatting Helpers ---

def format_structured_results(results: Dict[str, Any]) -> tuple:
    if "error" in results: return results["error"], "{}", "{}", "{}"
    success_count = sum(1 for r in results.get("results", []) if r.get("success"))
    summary_text = f"**Tests Run**: {len(results.get('results', []))}\n**Successful**: {success_count}"
    successful_results = [r for r in results.get("results", []) if r.get("success")]
    raw_json = json.dumps(results, indent=2, default=str)

    if not successful_results:
        return summary_text, "{}", "No data extracted.", raw_json

    best_result = successful_results[0]
    structured_data = json.dumps(best_result.get("data", {}), indent=2)
    quality = best_result.get("quality", {})
    quality_summary = "Quality assessment not available."
    if quality and quality.get("success"):
        assessment = quality["assessment"]
        quality_summary = (f"**Completeness**: {assessment.get('completeness_score', 0):.1f}/10\n"
                         f"**Accuracy**: {assessment.get('accuracy_score', 0):.1f}/10\n"
                         f"**Structure**: {assessment.get('structure_score', 0):.1f}/10\n"
                         f"**Confidence**: {str(assessment.get('confidence_level', 'unknown')).title()}\n\n"
                         f"**Recommendations**:\n- " + "\n- ".join(assessment.get('recommendations', [])))
    return summary_text, structured_data, quality_summary, raw_json

def format_traditional_results(results: Dict[str, Any]) -> tuple:
    if "error" in results: return results["error"], "", "{}"
    summary_text = f"**Models Tested**: {len(results.get('results', []))}"
    all_text_parts = []
    for res in results.get("results", []):
        if res.get("success"):
            part = (f"### {res['model'].upper()}\n"
                    f"**Confidence**: {res.get('confidence', 0):.2f} | **Time**: {res.get('execution_time', 0):.2f}s\n\n"
                    f"```\n{res.get('text', '')}\n```")
        else:
            part = f"### {res['model'].upper()}\n\n**❌ Error**: {res.get('error', 'Unknown error')}"
        all_text_parts.append(part)
    raw_json = json.dumps(results, indent=2, default=str)
    return summary_text, "\n\n---\n\n".join(all_text_parts), raw_json

# --- Main App Creation ---

def create_app():
    if not os.getenv("OPENROUTER_API_KEY"):
        with gr.Blocks(title="OCR Benchmark - Error") as app:
            gr.Markdown("# ⚠️ Configuration Error\n`OPENROUTER_API_KEY` not found in environment variables or `.env` file.")
        return app

    structured_handler = StandaloneStructuredOCR()
    traditional_handler = StandaloneTraditionalOCR()

    with gr.Blocks(title="🚀 Advanced OCR Benchmark Suite", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🚀 Advanced OCR Benchmark Suite")
        gr.Markdown("Upload an image or select one of the examples from the `data` directory using the upload button.")
        
        with gr.Tabs():
            with gr.Tab("📊 Structured JSON Extraction"):
                with gr.Row():
                    with gr.Column(scale=1):
                        struct_image = gr.Image(type="pil", label="🖼️ Upload Dashboard")
                        struct_models = gr.CheckboxGroup(choices=["gpt-4o", "openai/gpt-4o-mini", "openai/gpt-4.1", "openai/gpt-4.1-mini", "openai/gpt-4.1-nano", "anthropic/claude-3.5-sonnet", "anthropic/claude-sonnet-4", "anthropic/claude-3.7-sonnet", "google/gemini-2.5-pro", "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite-preview-06-17", "google/gemini-pro-1.5", "google/gemini-flash-1.5"], value=["gpt-4o"], label="🤖 Select VLM Models")
                        struct_extract_btn = gr.Button("Extract Structured Data", variant="primary")
                    with gr.Column(scale=2):
                        struct_summary = gr.Markdown(label="📝 Summary")
                        struct_data = gr.Code(label="📄 Structured JSON", language="json", lines=15)
                        struct_quality = gr.Markdown(label="⭐ Quality Assessment")
                with gr.Accordion("Raw JSON Output", open=False):
                    struct_raw = gr.Code(label="Complete Raw Results", language="json", lines=10)

                async def run_structured(image, models):
                    if image is None or not models: return "Provide an image and model.", "{}", "{}", "{}"
                    results = {"timestamp": datetime.now().isoformat(), "results": []}
                    for model in models:
                        extraction = await structured_handler.extract(image, model)
                        quality = {}
                        if extraction.get("success"):
                            quality = await structured_handler.assess(extraction["data"])
                        results["results"].append({"success": extraction.get("success"), "data": extraction.get("data"), "quality": quality, "error": extraction.get("error")})
                    return format_structured_results(results)

                struct_extract_btn.click(
                    fn=run_structured,
                    inputs=[struct_image, struct_models],
                    outputs=[struct_summary, struct_data, struct_quality, struct_raw]
                )

            with gr.Tab("⚙️ Traditional OCR"):
                with gr.Row():
                    with gr.Column(scale=1):
                        trad_image = gr.Image(type="pil", label="🖼️ Upload Image")
                        trad_models = gr.CheckboxGroup(choices=["easyocr", "paddleocr", "tesseract"], value=["easyocr", "paddleocr"], label="⚙️ Select OCR Engines")
                        trad_extract_btn = gr.Button("Extract Text", variant="secondary")
                    with gr.Column(scale=2):
                        trad_summary = gr.Markdown(label="📝 Summary")
                        trad_text = gr.Markdown(label="✍️ Extracted Text")
                with gr.Accordion("Raw JSON Output", open=False):
                    trad_raw = gr.Code(label="Complete Raw Results", language="json", lines=10)

                async def run_traditional(image, models):
                    if image is None or not models: return "Provide an image and model.", "", "{}"
                    tasks = [traditional_handler.extract_text(image, model) for model in models]
                    results_list = await asyncio.gather(*tasks)
                    return format_traditional_results({"results": results_list})

                trad_extract_btn.click(
                    fn=run_traditional,
                    inputs=[trad_image, trad_models],
                    outputs=[trad_summary, trad_text, trad_raw]
                )
    return app

if __name__ == "__main__":
    main_app = create_app()
    main_app.launch(server_name="127.0.0.1", server_port=7860, show_error=True, debug=True)