import streamlit as st
import asyncio
import pandas as pd
import numpy as np
from PIL import Image
import io
import zipfile
import json
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Import our custom modules
from src.config import config
from src.factory import OCRProviderFactory, di_container
from src.providers.base import OCRResult

class AdvancedOCRBenchmark:
    """Advanced OCR benchmarking application"""
    
    def __init__(self):
        self.config = config
        self.factory = OCRProviderFactory(self.config)
        self.results_history = []
    
    async def run_single_benchmark(
        self, 
        image: Image.Image, 
        provider, 
        model_id: str, 
        model_config: Dict,
        custom_prompt: str = None
    ) -> OCRResult:
        """Run OCR on single image with single model"""
        
        prompt = custom_prompt or self.config.default_ocr_prompt
        
        try:
            if hasattr(provider, 'extract_text'):
                if "traditional" in str(type(provider)):
                    result = await provider.extract_text(image, model_id)
                else:
                    result = await provider.extract_text(image, model_id, prompt)
                return result
            else:
                return OCRResult(
                    text="",
                    execution_time=0,
                    error="Provider doesn't support extract_text method",
                    provider=str(type(provider)),
                    model=model_id
                )
        except Exception as e:
            return OCRResult(
                text="",
                execution_time=0,
                error=str(e),
                provider=str(type(provider)),
                model=model_id
            )
    
    async def run_batch_benchmark(
        self, 
        images: List[Image.Image], 
        selected_models: List[str],
        custom_prompt: str = None
    ) -> List[Dict[str, Any]]:
        """Run comprehensive benchmark on multiple images and models"""
        
        benchmark_suite = self.factory.create_benchmark_suite(selected_models)
        all_results = []
        
        for i, image in enumerate(images):
            st.write(f"🔄 Processing image {i+1}/{len(images)}...")
            
            # Create progress bar for this image
            progress_bar = st.progress(0)
            
            for j, (provider, model_id, model_config) in enumerate(benchmark_suite):
                # Update progress
                progress = (j + 1) / len(benchmark_suite)
                progress_bar.progress(progress)
                
                # Show current model being tested
                st.write(f"   Testing: {model_config.get('name', model_id)}")
                
                # Run OCR
                result = await self.run_single_benchmark(
                    image, provider, model_id, model_config, custom_prompt
                )
                
                # Store result with metadata
                result_data = result.to_dict()
                result_data.update({
                    'image_index': i,
                    'image_name': f"image_{i+1}",
                    'timestamp': datetime.now().isoformat(),
                    'model_name': model_config.get('name', model_id)
                })
                
                all_results.append(result_data)
            
            # Clear progress bar
            progress_bar.empty()
        
        return all_results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze benchmark results and generate insights"""
        
        df = pd.DataFrame(results)
        
        if df.empty:
            return {"error": "No results to analyze"}
        
        analysis = {
            "summary": {
                "total_tests": len(df),
                "unique_models": df['model'].nunique(),
                "unique_images": df['image_index'].nunique(),
                "success_rate": (df['error'].isna().sum() / len(df)) * 100,
                "average_execution_time": df[df['error'].isna()]['execution_time'].mean(),
                "total_cost": df[df['error'].isna()]['cost'].sum()
            },
            "model_performance": {},
            "image_difficulty": {},
            "cost_analysis": {}
        }
        
        # Model performance analysis
        model_stats = df.groupby('model').agg({
            'execution_time': ['mean', 'std'],
            'confidence': ['mean', 'std'],
            'character_count': ['mean', 'std'],
            'cost': 'sum',
            'error': lambda x: x.isna().sum() / len(x) * 100  # success rate
        }).round(4)
        
        analysis["model_performance"] = model_stats.to_dict()
        
        # Image difficulty analysis (based on processing time and success rates)
        image_stats = df.groupby('image_index').agg({
            'execution_time': 'mean',
            'character_count': 'mean',
            'error': lambda x: x.isna().sum() / len(x) * 100
        }).round(4)
        
        analysis["image_difficulty"] = image_stats.to_dict()
        
        # Cost analysis
        cost_by_provider = df.groupby('provider')['cost'].sum().to_dict()
        analysis["cost_analysis"] = cost_by_provider
        
        return analysis

def main():
    st.set_page_config(
        page_title="Advanced OCR Benchmark Suite",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 Advanced OCR Benchmark Suite")
    st.markdown("**Comprehensive testing of Traditional OCR vs Modern VLM models**")
    
    # Check API key configuration
    if not config.openrouter_api_key:
        st.error("⚠️ OpenRouter API key not configured. Create `.env` file with OPENROUTER_API_KEY")
        st.info("Copy `.env.example` to `.env` and add your API key")
        return
    
    # Initialize benchmark app
    benchmark_app = AdvancedOCRBenchmark()
    
    # Sidebar configuration
    st.sidebar.header("🔧 Benchmark Configuration")
    
    # Model selection
    available_models = benchmark_app.factory.get_available_models()
    
    st.sidebar.subheader("📊 Select Models to Test")
    
    # Group models by category
    vlm_models = {k: v for k, v in available_models.items() if "vlm_" in k}
    traditional_models = {k: v for k, v in available_models.items() if "traditional_" in k}
    
    selected_models = []
    
    # VLM Models
    if vlm_models:
        st.sidebar.write("**🤖 AI Vision Language Models:**")
        for model_key, model_info in vlm_models.items():
            if st.sidebar.checkbox(
                f"{model_info['name']} (${model_info.get('cost_per_1k_tokens', 0):.4f}/1k tokens)",
                key=f"select_{model_key}"
            ):
                selected_models.append(model_key)
    
    # Traditional Models  
    if traditional_models:
        st.sidebar.write("**🔧 Traditional OCR Models:**")
        for model_key, model_info in traditional_models.items():
            if st.sidebar.checkbox(
                f"{model_info['name']} (Free)",
                key=f"select_{model_key}",
                value=True  # Select traditional models by default
            ):
                selected_models.append(model_key)
    
    # Custom prompt
    st.sidebar.subheader("📝 Custom OCR Prompt")
    custom_prompt = st.sidebar.text_area(
        "OCR Instructions:",
        value=config.default_ocr_prompt,
        height=100,
        help="Customize the prompt sent to AI models"
    )
    
    # File upload
    st.sidebar.subheader("📁 Upload Images")
    uploaded_files = st.sidebar.file_uploader(
        "Select images for benchmarking",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        accept_multiple_files=True,
        help="Upload multiple images to test OCR accuracy across different content types"
    )
    
    # Main content area
    if uploaded_files and selected_models:
        
        # Display uploaded images
        st.subheader("📷 Images to Process")
        cols = st.columns(min(len(uploaded_files), 4))
        images = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            images.append(image)
            
            with cols[i % 4]:
                st.image(image, caption=f"Image {i+1}", use_column_width=True)
        
        # Show selected models
        st.subheader("🎯 Selected Models")
        model_info_df = pd.DataFrame([
            {
                "Model": available_models[model_key]["name"],
                "Provider": available_models[model_key].get("provider", "N/A"),
                "Cost/1k tokens": f"${available_models[model_key].get('cost_per_1k_tokens', 0):.4f}"
            }
            for model_key in selected_models
        ])
        st.dataframe(model_info_df, use_container_width=True)
        
        # Run benchmark button
        if st.button("🚀 Start Comprehensive Benchmark", type="primary"):
            
            with st.spinner("Running comprehensive OCR benchmark..."):
                
                # Run async benchmark
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    results = loop.run_until_complete(
                        benchmark_app.run_batch_benchmark(
                            images, selected_models, custom_prompt
                        )
                    )
                    
                    loop.close()
                    
                    if results:
                        st.success(f"✅ Benchmark completed! Processed {len(results)} tests.")
                        
                        # Store results in session state
                        st.session_state['benchmark_results'] = results
                        
                        # Analyze results
                        analysis = benchmark_app.analyze_results(results)
                        st.session_state['benchmark_analysis'] = analysis
                        
                    else:
                        st.error("❌ No results generated")
                        
                except Exception as e:
                    st.error(f"❌ Benchmark failed: {str(e)}")
    
    # Display results if available
    if 'benchmark_results' in st.session_state:
        
        results = st.session_state['benchmark_results']
        analysis = st.session_state['benchmark_analysis']
        
        st.header("📊 Benchmark Results")
        
        # Summary metrics
        summary = analysis["summary"]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Success Rate", f"{summary['success_rate']:.1f}%")
        with col2:
            st.metric("Avg Time", f"{summary['average_execution_time']:.2f}s")
        with col3:
            st.metric("Total Tests", summary['total_tests'])
        with col4:
            st.metric("Total Cost", f"${summary['total_cost']:.4f}")
        
        # Results table
        st.subheader("📋 Detailed Results")
        results_df = pd.DataFrame(results)
        
        # Filter successful results for display
        success_df = results_df[results_df['error'].isna()].copy()
        
        if not success_df.empty:
            display_df = success_df[[
                'image_name', 'model_name', 'execution_time', 
                'confidence', 'character_count', 'cost'
            ]].copy()
            
            display_df['execution_time'] = display_df['execution_time'].round(3)
            display_df['confidence'] = display_df['confidence'].round(3)
            display_df['cost'] = display_df['cost'].round(6)
            
            st.dataframe(display_df, use_container_width=True)
        
        # Performance charts
        st.subheader("📈 Performance Analysis")
        
        if not success_df.empty:
            
            # Execution time comparison
            fig_time = px.box(
                success_df, 
                x='model_name', 
                y='execution_time',
                title="Execution Time by Model"
            )
            fig_time.update_xaxes(tickangle=45)
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Character count comparison
            fig_chars = px.box(
                success_df,
                x='model_name',
                y='character_count', 
                title="Characters Extracted by Model"
            )
            fig_chars.update_xaxes(tickangle=45)
            st.plotly_chart(fig_chars, use_container_width=True)
            
            # Cost analysis
            cost_by_model = success_df.groupby('model_name')['cost'].sum().reset_index()
            fig_cost = px.bar(
                cost_by_model,
                x='model_name',
                y='cost',
                title="Total Cost by Model"
            )
            fig_cost.update_xaxes(tickangle=45)
            st.plotly_chart(fig_cost, use_container_width=True)
        
        # Export results
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON export
            json_data = {
                "results": results,
                "analysis": analysis,
                "config": {
                    "custom_prompt": custom_prompt,
                    "selected_models": selected_models,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            json_str = json.dumps(json_data, indent=2)
            st.download_button(
                label="📄 Download JSON Report",
                data=json_str,
                file_name=f"ocr_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # CSV export
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="📊 Download CSV Data", 
                data=csv_data,
                file_name=f"ocr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    else:
        # Instructions when no results
        st.info("👆 Configure models and upload images to start benchmarking")
        
        st.subheader("💡 How to Use")
        st.markdown("""
        1. **Configure API Key**: Copy `.env.example` to `.env` and add your OpenRouter API key
        2. **Select Models**: Choose which OCR models to benchmark (AI vs Traditional)
        3. **Upload Images**: Add multiple test images (documents, screenshots, etc.)
        4. **Customize Prompt**: Modify OCR instructions for AI models
        5. **Run Benchmark**: Start comprehensive testing across all combinations
        6. **Analyze Results**: View performance metrics, charts, and export data
        
        ### 🎯 Supported Models:
        - **GPT-4 Vision & GPT-4o**: OpenAI's vision models
        - **Claude 3.5 Sonnet/Haiku**: Anthropic's vision models  
        - **Gemini Pro/Flash 1.5**: Google's multimodal models
        - **Mistral Pixtral**: Mistral's vision model
        - **Qwen2-VL**: Alibaba's vision language models
        - **EasyOCR, PaddleOCR, Tesseract**: Traditional OCR engines
        """)

if __name__ == "__main__":
    main()