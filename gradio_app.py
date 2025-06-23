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

# Import our structured OCR modules
from src.providers.traditional_provider import TraditionalOCRProvider
# Temporarily disable structured provider to avoid Gradio/Pydantic schema conflicts
STRUCTURED_AVAILABLE = False
StructuredOCRProvider = None

class SimpleConfig:
    """Minimal config to avoid Pydantic issues"""
    def __init__(self):
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

class GradioOCRBenchmark:
    """Modern Gradio interface for OCR benchmarking"""
    
    def __init__(self):
        self.config = SimpleConfig()
        self.structured_provider = None
        self.traditional_provider = None
        
    def init_providers(self):
        """Initialize providers with error handling"""
        try:
            if STRUCTURED_AVAILABLE and self.config.openrouter_api_key and StructuredOCRProvider:
                self.structured_provider = StructuredOCRProvider(self.config)
            self.traditional_provider = TraditionalOCRProvider(self.config)
            return True
        except Exception as e:
            return f"Provider initialization failed: {str(e)}"
    
    async def run_single_extraction(
        self, 
        image: Image.Image, 
        models: List[str],
        extraction_type: str = "structured"
    ) -> Dict[str, Any]:
        """Run OCR extraction on single image with selected models"""
        
        if not image:
            return {"error": "No image provided"}
        
        if not models:
            return {"error": "No models selected"}
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "extraction_type": extraction_type,
            "results": [],
            "summary": {}
        }
        
        for model in models:
            try:
                if extraction_type == "structured" and self.structured_provider:
                    # Structured JSON extraction
                    extraction = await self.structured_provider.extract_structured_data(image, model)
                    
                    if extraction["success"]:
                        # Quality assessment
                        quality = await self.structured_provider.assess_extraction_quality(
                            extraction["data"],
                            f"Dashboard image analysis with {model}",
                            "gpt-4o-mini"
                        )
                        
                        results["results"].append({
                            "model": model,
                            "type": "structured",
                            "success": True,
                            "data": extraction["data"],
                            "quality": quality.get("assessment") if quality.get("success") else None,
                            "usage": extraction.get("usage", {})
                        })
                    else:
                        results["results"].append({
                            "model": model,
                            "type": "structured", 
                            "success": False,
                            "error": extraction.get("error", "Unknown error")
                        })
                
                elif extraction_type == "traditional" and self.traditional_provider:
                    # Traditional OCR
                    result = await self.traditional_provider.extract_text(image, model)
                    
                    results["results"].append({
                        "model": model,
                        "type": "traditional",
                        "success": not bool(result.error),
                        "text": result.text,
                        "confidence": result.confidence,
                        "execution_time": result.execution_time,
                        "error": result.error
                    })
                    
            except Exception as e:
                results["results"].append({
                    "model": model,
                    "type": extraction_type,
                    "success": False,
                    "error": str(e)
                })
        
        # Generate summary
        successful = [r for r in results["results"] if r["success"]]
        results["summary"] = {
            "total_tests": len(results["results"]),
            "successful": len(successful),
            "success_rate": len(successful) / len(results["results"]) * 100 if results["results"] else 0
        }
        
        return results
    
    def format_structured_results(self, results: Dict[str, Any]) -> tuple:
        """Format structured results for Gradio display"""
        
        if "error" in results:
            return results["error"], "{}", "{}", "{}"
        
        summary_text = f"""
## 📊 Benchmark Summary
- **Tests Run**: {results['summary']['total_tests']}
- **Successful**: {results['summary']['successful']}
- **Success Rate**: {results['summary']['success_rate']:.1f}%
        """
        
        # Best result for detailed display
        successful_results = [r for r in results["results"] if r["success"]]
        
        if not successful_results:
            return summary_text, "{}", "{}", "{}"
        
        best_result = successful_results[0]  # Take first successful result
        
        # Format structured data
        structured_data = json.dumps(best_result.get("data", {}), indent=2)
        
        # Format quality assessment
        quality = best_result.get("quality", {})
        if quality:
            quality_summary = f"""
**Completeness**: {quality.get('completeness_score', 0):.1f}/10
**Accuracy**: {quality.get('accuracy_score', 0):.1f}/10  
**Structure**: {quality.get('structure_score', 0):.1f}/10
**Confidence**: {quality.get('confidence_level', 'unknown').title()}

**Recommendations**:
{chr(10).join(f"• {rec}" for rec in quality.get('recommendations', []))}
            """
        else:
            quality_summary = "Quality assessment not available"
        
        # All results summary
        all_results = json.dumps(results, indent=2, default=str)
        
        return summary_text, structured_data, quality_summary, all_results
    
    def format_traditional_results(self, results: Dict[str, Any]) -> tuple:
        """Format traditional OCR results for display"""
        
        if "error" in results:
            return results["error"], "{}"
        
        summary_text = f"""
## 📊 Traditional OCR Results  
- **Models Tested**: {len(results['results'])}
- **Success Rate**: {results['summary']['success_rate']:.1f}%
        """
        
        # Combine all extracted text
        all_text = []
        for result in results["results"]:
            if result["success"]:
                model_name = result["model"]
                text = result.get("text", "")
                confidence = result.get("confidence", 0)
                time = result.get("execution_time", 0)
                
                all_text.append(f"""
### {model_name.upper()}
**Confidence**: {confidence:.2f} | **Time**: {time:.2f}s

{text}

---
                """)
        
        combined_text = "\n".join(all_text) if all_text else "No text extracted"
        all_results = json.dumps(results, indent=2, default=str)
        
        return summary_text, combined_text, all_results
    
    def discover_sample_images(self) -> List[str]:
        """Get sample images from data directory"""
        if self.structured_provider:
            return self.structured_provider.discover_data_files("data")
        return []

def create_app():
    """Create and configure Gradio interface"""
    
    benchmark = GradioOCRBenchmark()
    init_result = benchmark.init_providers()
    
    if init_result != True:
        # Error in initialization
        def error_interface():
            return gr.Markdown(f"""
            # ❌ Configuration Error
            
            {init_result}
            
            ## Quick Fix:
            1. Copy `.env.example` to `.env`
            2. Add your OpenRouter API key: `OPENROUTER_API_KEY=sk-or-v1-your-key`
            3. Restart the app
            """)
        
        return gr.Interface(
            fn=lambda: None,
            inputs=[],
            outputs=error_interface(),
            title="OCR Benchmark - Configuration Error"
        )
    
    # Main interface
    with gr.Blocks(
        title="🔬 Advanced OCR Benchmark Suite",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        """
    ) as app:
        
        gr.Markdown("""
        # 🔬 Advanced OCR Benchmark Suite
        
        Compare **Traditional OCR** vs **Modern Vision Language Models** with structured JSON extraction and quality assessment.
        """)
        
        # Sample images discovery
        sample_images = benchmark.discover_sample_images()
        if sample_images:
            gr.Markdown(f"""
            ### 📁 **{len(sample_images)} Sample Images Detected:**
            {', '.join([os.path.basename(img) for img in sample_images])}
            """)
        
        with gr.Tabs():
            
            # TAB 1: Structured JSON Extraction (MAIN)
            with gr.Tab("🔬 Structured JSON Extraction", elem_id="structured_tab"):
                gr.Markdown("""
                ### Extract structured data from dashboard images with AI quality assessment
                **Best for**: Business dashboards, analytics screenshots, complex layouts
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        struct_image = gr.Image(
                            type="pil", 
                            label="📷 Upload Dashboard Image",
                            height=300
                        )
                        
                        struct_models = gr.CheckboxGroup(
                            choices=[
                                "gpt-4o",
                                "claude-3-5-sonnet-20241022", 
                                "gpt-4-vision-preview",
                                "claude-3-5-haiku-20241022",
                                "google/gemini-pro-1.5",
                                "google/gemini-flash-1.5",
                                "mistralai/pixtral-12b",
                                "qwen/qwen-2-vl-72b-instruct"
                            ],
                            value=["gpt-4o", "claude-3-5-sonnet-20241022"],
                            label="🤖 Select VLM Models",
                            info="Choose models for comparison"
                        )
                        
                        struct_extract_btn = gr.Button(
                            "🚀 Extract Structured Data", 
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        struct_summary = gr.Markdown(label="📊 Summary")
                        
                        with gr.Row():
                            struct_data = gr.Code(
                                label="📋 Structured JSON Data",
                                language="json",
                                lines=15
                            )
                            struct_quality = gr.Markdown(label="🎯 Quality Assessment")
                
                with gr.Accordion("🔍 Raw Results (JSON)", open=False):
                    struct_raw = gr.Code(
                        label="Complete Results",
                        language="json", 
                        lines=10
                    )
                
                # Structured extraction event
                async def run_structured_extraction(image, models):
                    if not image or not models:
                        return "❌ Please provide an image and select at least one model", "{}", "No assessment", "{}"
                    
                    results = await benchmark.run_single_extraction(image, models, "structured")
                    return benchmark.format_structured_results(results)
                
                struct_extract_btn.click(
                    fn=run_structured_extraction,
                    inputs=[struct_image, struct_models],
                    outputs=[struct_summary, struct_data, struct_quality, struct_raw]
                )
            
            # TAB 2: Traditional OCR Comparison  
            with gr.Tab("🔧 Traditional OCR", elem_id="traditional_tab"):
                gr.Markdown("""
                ### Compare traditional OCR engines (EasyOCR, PaddleOCR, Tesseract)
                **Best for**: Simple text extraction, speed comparison
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        trad_image = gr.Image(
                            type="pil",
                            label="📷 Upload Image", 
                            height=300
                        )
                        
                        trad_models = gr.CheckboxGroup(
                            choices=["easyocr", "paddleocr", "tesseract"],
                            value=["easyocr", "paddleocr"],
                            label="🔧 Select Traditional OCR",
                            info="Choose OCR engines to compare"
                        )
                        
                        trad_extract_btn = gr.Button(
                            "📝 Extract Text",
                            variant="secondary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        trad_summary = gr.Markdown(label="📊 Summary")
                        trad_text = gr.Textbox(
                            label="📝 Extracted Text",
                            lines=20,
                            max_lines=25
                        )
                
                with gr.Accordion("🔍 Raw Results (JSON)", open=False):
                    trad_raw = gr.Code(
                        label="Complete Results",
                        language="json",
                        lines=10
                    )
                
                # Traditional extraction event
                async def run_traditional_extraction(image, models):
                    if not image or not models:
                        return "❌ Please provide an image and select at least one model", "", "{}"
                    
                    results = await benchmark.run_single_extraction(image, models, "traditional")
                    return benchmark.format_traditional_results(results)
                
                trad_extract_btn.click(
                    fn=run_traditional_extraction,
                    inputs=[trad_image, trad_models],
                    outputs=[trad_summary, trad_text, trad_raw]
                )
            
            # TAB 3: Batch Processing
            with gr.Tab("📁 Batch Processing", elem_id="batch_tab"):
                gr.Markdown("""
                ### Process multiple images automatically
                **Best for**: Dataset evaluation, comprehensive benchmarking
                """)
                
                with gr.Row():
                    with gr.Column():
                        batch_models = gr.CheckboxGroup(
                            choices=["gpt-4o", "claude-3-5-sonnet-20241022", "google/gemini-flash-1.5"],
                            value=["gpt-4o"],
                            label="🤖 Select Models for Batch Processing"
                        )
                        
                        batch_btn = gr.Button(
                            "🚀 Process All Images in data/",
                            variant="primary"
                        )
                        
                        batch_status = gr.Markdown("Ready to process")
                    
                    with gr.Column():
                        batch_results = gr.Code(
                            label="📊 Batch Results",
                            language="json",
                            lines=20
                        )
                
                # Batch processing event
                async def run_batch_processing(models):
                    if not models:
                        return "❌ Select at least one model", "{}"
                    
                    if not benchmark.structured_provider:
                        return "❌ Structured provider not available", "{}"
                    
                    try:
                        results = await benchmark.structured_provider.run_comprehensive_benchmark(
                            data_dir="data",
                            models=models
                        )
                        
                        summary = results.get("summary", {})
                        status = f"""
                        ✅ **Batch processing completed!**
                        - Images processed: {summary.get('total_images', 0)}
                        - Models tested: {len(summary.get('models_tested', []))}
                        - Results generated: {len(results.get('results', []))}
                        """
                        
                        return status, json.dumps(results, indent=2, default=str)
                        
                    except Exception as e:
                        return f"❌ Batch processing failed: {str(e)}", "{}"
                
                batch_btn.click(
                    fn=run_batch_processing,
                    inputs=[batch_models],
                    outputs=[batch_status, batch_results]
                )
        
        # Footer
        gr.Markdown("""
        ---
        ### 💡 **Quick Start Guide**
        1. **Structured Extraction**: Upload dashboard → Select VLM models → Get JSON + quality scores
        2. **Traditional OCR**: Upload any image → Compare EasyOCR/PaddleOCR/Tesseract speeds
        3. **Batch Processing**: Process all images in `data/` folder automatically
        
        ### 🔧 **Setup**
        ```bash
        cp .env.example .env
        # Edit .env: OPENROUTER_API_KEY=your_key
        poetry install && poetry run python gradio_app.py
        ```
        """)
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,  # Enable public sharing for WSL compatibility
        show_error=True,
        debug=True
    )