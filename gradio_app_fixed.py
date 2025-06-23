#!/usr/bin/env python3
"""
Fixed Gradio app for OCR benchmarking - avoiding Pydantic schema conflicts
"""

import gradio as gr
import asyncio
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import os
import time

# Traditional OCR imports with error handling
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

class SimpleOCRResult:
    """Simple OCR result without Pydantic"""
    def __init__(self, text: str, confidence: float = 0.0, execution_time: float = 0.0, 
                 provider: str = "", error: Optional[str] = None):
        self.text = text
        self.confidence = confidence
        self.execution_time = execution_time
        self.provider = provider
        self.error = error

class TraditionalOCRProvider:
    """Traditional OCR provider without Pydantic dependencies"""
    
    def __init__(self):
        self.easyocr_reader = None
        self.paddleocr_reader = None
        
        # Initialize EasyOCR
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'])
            except Exception as e:
                print(f"EasyOCR initialization failed: {e}")
        
        # Initialize PaddleOCR
        if PADDLEOCR_AVAILABLE:
            try:
                self.paddleocr_reader = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            except Exception as e:
                print(f"PaddleOCR initialization failed: {e}")
    
    async def extract_with_easyocr(self, image: Image.Image) -> SimpleOCRResult:
        """Extract text using EasyOCR"""
        if not self.easyocr_reader:
            return SimpleOCRResult("", 0.0, 0.0, "easyocr", "EasyOCR not available")
        
        try:
            start_time = time.time()
            # Convert PIL to numpy array
            import numpy as np
            img_array = np.array(image)
            
            # Run EasyOCR
            results = self.easyocr_reader.readtext(img_array)
            execution_time = time.time() - start_time
            
            # Extract text and calculate average confidence
            text_parts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                text_parts.append(text)
                confidences.append(confidence)
            
            combined_text = " ".join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return SimpleOCRResult(combined_text, avg_confidence, execution_time, "easyocr")
        
        except Exception as e:
            return SimpleOCRResult("", 0.0, 0.0, "easyocr", str(e))
    
    async def extract_with_paddleocr(self, image: Image.Image) -> SimpleOCRResult:
        """Extract text using PaddleOCR"""
        if not self.paddleocr_reader:
            return SimpleOCRResult("", 0.0, 0.0, "paddleocr", "PaddleOCR not available")
        
        try:
            start_time = time.time()
            # Convert PIL to numpy array
            import numpy as np
            img_array = np.array(image)
            
            # Run PaddleOCR
            results = self.paddleocr_reader.ocr(img_array, cls=True)
            execution_time = time.time() - start_time
            
            # Extract text and calculate average confidence
            text_parts = []
            confidences = []
            
            if results and results[0]:
                for line in results[0]:
                    if len(line) >= 2:
                        text = line[1][0]
                        confidence = line[1][1]
                        text_parts.append(text)
                        confidences.append(confidence)
            
            combined_text = " ".join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return SimpleOCRResult(combined_text, avg_confidence, execution_time, "paddleocr")
        
        except Exception as e:
            return SimpleOCRResult("", 0.0, 0.0, "paddleocr", str(e))
    
    async def extract_with_tesseract(self, image: Image.Image) -> SimpleOCRResult:
        """Extract text using Tesseract"""
        if not TESSERACT_AVAILABLE:
            return SimpleOCRResult("", 0.0, 0.0, "tesseract", "Tesseract not available")
        
        try:
            start_time = time.time()
            
            # Run Tesseract
            text = pytesseract.image_to_string(image)
            
            # Get confidence data
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            execution_time = time.time() - start_time
            
            return SimpleOCRResult(text.strip(), avg_confidence / 100.0, execution_time, "tesseract")
        
        except Exception as e:
            return SimpleOCRResult("", 0.0, 0.0, "tesseract", str(e))

class GradioOCRApp:
    """Fixed Gradio OCR application"""
    
    def __init__(self):
        self.traditional_provider = TraditionalOCRProvider()
    
    async def run_traditional_ocr(self, image: Image.Image, selected_models: List[str]) -> Dict[str, Any]:
        """Run traditional OCR with selected models"""
        if not image:
            return {"error": "No image provided"}
        
        if not selected_models:
            return {"error": "No models selected"}
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "results": []
        }
        
        for model in selected_models:
            if model == "easyocr":
                result = await self.traditional_provider.extract_with_easyocr(image)
            elif model == "paddleocr":
                result = await self.traditional_provider.extract_with_paddleocr(image)
            elif model == "tesseract":
                result = await self.traditional_provider.extract_with_tesseract(image)
            else:
                continue
            
            results["results"].append({
                "model": model,
                "success": not bool(result.error),
                "text": result.text,
                "confidence": result.confidence,
                "execution_time": result.execution_time,
                "error": result.error
            })
        
        return results
    
    def format_results(self, results: Dict[str, Any]) -> tuple:
        """Format results for Gradio display"""
        if "error" in results:
            return results["error"], "", "{}"
        
        summary_text = f"""
## 📊 OCR Results Summary
- **Models Tested**: {len(results['results'])}
- **Timestamp**: {results['timestamp']}
        """
        
        # Combine all extracted text
        all_text = []
        for result in results["results"]:
            if result["success"]:
                model_name = result["model"].upper()
                text = result.get("text", "")
                confidence = result.get("confidence", 0)
                time_taken = result.get("execution_time", 0)
                
                all_text.append(f"""
### {model_name}
**Confidence**: {confidence:.2f} | **Time**: {time_taken:.2f}s

{text}

---
                """)
            else:
                model_name = result["model"].upper()
                error = result.get("error", "Unknown error")
                all_text.append(f"""
### {model_name}
❌ **Error**: {error}

---
                """)
        
        combined_text = "\n".join(all_text) if all_text else "No results"
        raw_json = json.dumps(results, indent=2, default=str)
        
        return summary_text, combined_text, raw_json

def create_app():
    """Create the Gradio app"""
    app_instance = GradioOCRApp()
    
    with gr.Blocks(
        title="🔧 Traditional OCR Benchmark",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        """
    ) as app:
        
        gr.Markdown("""
        # 🔧 Traditional OCR Benchmark
        
        Compare traditional OCR engines: **EasyOCR**, **PaddleOCR**, and **Tesseract**
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="📷 Upload Image",
                    height=300
                )
                
                model_selection = gr.CheckboxGroup(
                    choices=["easyocr", "paddleocr", "tesseract"],
                    value=["easyocr", "paddleocr"],
                    label="🔧 Select OCR Engines",
                    info="Choose OCR engines to compare"
                )
                
                extract_btn = gr.Button(
                    "📝 Extract Text",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                summary_output = gr.Markdown(label="📊 Summary")
                text_output = gr.Markdown(
                    label="📝 Extracted Text",
                    value="Upload an image and select OCR engines to get started"
                )
        
        with gr.Accordion("🔍 Raw Results (JSON)", open=False):
            json_output = gr.Code(
                label="Complete Results",
                language="json",
                lines=10
            )
        
        # Event handler
        async def process_ocr(image, models):
            if not image or not models:
                return "❌ Please provide an image and select at least one OCR engine", "No results", "{}"
            
            results = await app_instance.run_traditional_ocr(image, models)
            return app_instance.format_results(results)
        
        extract_btn.click(
            fn=process_ocr,
            inputs=[image_input, model_selection],
            outputs=[summary_output, text_output, json_output]
        )
        
        # Footer
        gr.Markdown("""
        ---
        ### 💡 **How to Use**
        1. Upload an image with text
        2. Select one or more OCR engines 
        3. Click "Extract Text" to compare results
        
        ### 🔧 **OCR Engines**
        - **EasyOCR**: Deep learning-based OCR with good multilingual support
        - **PaddleOCR**: Efficient OCR from PaddlePaddle with layout analysis
        - **Tesseract**: Traditional OCR engine from Google
        """)
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        show_error=True,
        debug=True
    )