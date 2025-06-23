# Structured Benchmark - Standalone Class (converted from Streamlit)
import asyncio
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
import os
from pathlib import Path

# Import our structured OCR modules
from src.config import config
from src.providers.structured_provider import StructuredOCRProvider
from src.schemas import DashboardData, QualityAssessment

class StructuredBenchmarkApp:
    """Streamlit app for structured JSON extraction benchmark"""
    
    def __init__(self):
        self.config = config
        self.provider = None
        
    def init_provider(self):
        """Initialize structured provider with error handling"""
        try:
            if not self.config.openrouter_api_key:
                print("⚠️ OpenRouter API key not configured!")
                print("Create `.env` file with: `OPENROUTER_API_KEY=your_key_here`")
                return False
            
            self.provider = StructuredOCRProvider(self.config)
            return True
        except Exception as e:
            print(f"❌ Provider initialization failed: {str(e)}")
            return False
    
    def display_dashboard_data(self, data: Dict[str, Any], title: str = "Extracted Data"):
        """Display structured dashboard data in organized format"""
        
        st.subheader(f"📊 {title}")
        
        # Dashboard title
        if data.get("dashboard_title"):
            st.write(f"**Dashboard Title:** {data['dashboard_title']}")
        
        # Key Metrics
        metrics = data.get("metrics", [])
        if metrics:
            st.write("**📈 Key Metrics:**")
            cols = st.columns(min(len(metrics), 4))
            for i, metric in enumerate(metrics):
                with cols[i % 4]:
                    value = f"{metric['value']}{metric.get('units', '')}"
                    st.metric(
                        label=metric['label'], 
                        value=value,
                        delta=metric.get('secondary_value')
                    )
        
        # Charts
        charts = data.get("charts", [])
        if charts:
            st.write("**📊 Charts & Visualizations:**")
            for i, chart in enumerate(charts):
                with st.expander(f"Chart {i+1}: {chart.get('title', 'Untitled')} ({chart.get('type', 'unknown')})"):
                    
                    # Chart metadata
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Type:** {chart.get('type', 'N/A')}")
                        st.write(f"**Data Points:** {len(chart.get('data_points', []))}")
                    with col2:
                        if chart.get('units'):
                            st.write(f"**Units:** {chart['units']}")
                        if chart.get('x_axis_label'):
                            st.write(f"**X-Axis:** {chart['x_axis_label']}")
                        if chart.get('y_axis_label'):
                            st.write(f"**Y-Axis:** {chart['y_axis_label']}")
                    
                    # Data points table
                    data_points = chart.get('data_points', [])
                    if data_points:
                        df_points = pd.DataFrame(data_points)
                        st.dataframe(df_points, use_container_width=True)
                        
                        # Simple visualization if numeric data
                        try:
                            if 'value' in df_points.columns and df_points['value'].dtype in ['int64', 'float64']:
                                if chart.get('type') == 'pie':
                                    fig = px.pie(df_points, values='value', names='label', title=chart.get('title', 'Chart'))
                                else:
                                    fig = px.bar(df_points, x='label', y='value', title=chart.get('title', 'Chart'))
                                st.plotly_chart(fig, use_container_width=True)
                        except:
                            pass  # Skip visualization if data format doesn't work
        
        # Time Series
        time_series = data.get("time_series", [])
        if time_series:
            st.write("**📈 Time Series Data:**")
            for i, ts in enumerate(time_series):
                with st.expander(f"Time Series {i+1}: {ts.get('title', 'Untitled')}"):
                    ts_data = ts.get('data', [])
                    if ts_data:
                        df_ts = pd.DataFrame(ts_data)
                        st.dataframe(df_ts, use_container_width=True)
                        
                        # Time series plot
                        try:
                            fig = px.line(df_ts, x='period', y='value', title=ts.get('title', 'Time Series'))
                            st.plotly_chart(fig, use_container_width=True)
                        except:
                            pass
        
        # Other text content
        text_content = data.get("text_content", [])
        if text_content:
            st.write("**📝 Other Text Content:**")
            for text in text_content:
                st.write(f"- {text}")
    
    def display_quality_assessment(self, assessment: Dict[str, Any]):
        """Display LLM quality assessment"""
        
        st.subheader("🎯 Quality Assessment")
        
        # Score metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Completeness", f"{assessment['completeness_score']:.1f}/10")
        with col2:
            st.metric("Accuracy", f"{assessment['accuracy_score']:.1f}/10")
        with col3:
            st.metric("Structure", f"{assessment['structure_score']:.1f}/10")
        with col4:
            confidence_color = {
                "high": "🟢",
                "medium": "🟡", 
                "low": "🔴"
            }.get(assessment.get('confidence_level', 'medium'), "⚪")
            st.metric("Confidence", f"{confidence_color} {assessment.get('confidence_level', 'N/A').title()}")
        
        # Overall score
        overall_score = (assessment['completeness_score'] + assessment['accuracy_score'] + assessment['structure_score']) / 3
        st.progress(overall_score / 10)
        st.write(f"**Overall Score: {overall_score:.1f}/10**")
        
        # Detailed feedback
        col1, col2 = st.columns(2)
        
        with col1:
            missing = assessment.get('missing_elements', [])
            if missing:
                st.write("**❌ Missing Elements:**")
                for item in missing:
                    st.write(f"- {item}")
            
            errors = assessment.get('potential_errors', [])
            if errors:
                st.write("**⚠️ Potential Errors:**")
                for error in errors:
                    st.write(f"- {error}")
        
        with col2:
            recommendations = assessment.get('recommendations', [])
            if recommendations:
                st.write("**💡 Recommendations:**")
                for rec in recommendations:
                    st.write(f"- {rec}")

def main():
    st.set_page_config(
        page_title="Structured OCR Benchmark",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 Structured Dashboard OCR Benchmark")
    st.markdown("**Extract structured JSON data from dashboard images with AI quality assessment**")
    
    # Initialize app
    app = StructuredBenchmarkApp()
    
    if not app.init_provider():
        return
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # Auto-discover images
    image_files = app.provider.discover_data_files("data")
    
    if not image_files:
        st.error("📁 No images found in `data/` directory")
        st.info("Add PNG/JPG images to the `data/` folder")
        return
    
    st.sidebar.success(f"📷 Found {len(image_files)} images")
    for img_file in image_files:
        st.sidebar.write(f"- {os.path.basename(img_file)}")
    
    # Model selection
    st.sidebar.subheader("🤖 Select Models")
    available_models = [
        "gpt-4o",
        "claude-3-5-sonnet-20241022", 
        "gpt-4-vision-preview",
        "claude-3-5-haiku-20241022",
        "google/gemini-pro-1.5"
    ]
    
    selected_models = []
    for model in available_models:
        if st.sidebar.checkbox(model, value=model in ["gpt-4o", "claude-3-5-sonnet-20241022"]):
            selected_models.append(model)
    
    if not selected_models:
        st.warning("⚠️ Select at least one model to test")
        return
    
    # Run benchmark
    if st.sidebar.button("🚀 Run Structured Benchmark", type="primary"):
        
        with st.spinner("Running comprehensive structured extraction..."):
            try:
                # Run async benchmark
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                results = loop.run_until_complete(
                    app.provider.run_comprehensive_benchmark(
                        data_dir="data",
                        models=selected_models
                    )
                )
                
                loop.close()
                
                # Store results in session state
                st.session_state['structured_results'] = results
                st.success("✅ Benchmark completed!")
                
            except Exception as e:
                st.error(f"❌ Benchmark failed: {str(e)}")
    
    # Display results
    if 'structured_results' in st.session_state:
        results = st.session_state['structured_results']
        
        # Summary
        summary = results.get("summary", {})
        st.header("📊 Benchmark Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Images Processed", summary.get("total_images", 0))
        with col2:
            st.metric("Models Tested", len(summary.get("models_tested", [])))
        with col3:
            successful_extractions = sum(1 for r in results.get("results", []) if r.get("extraction_result", {}).get("success"))
            st.metric("Successful Extractions", successful_extractions)
        
        # Detailed results
        for result in results.get("results", []):
            image_name = result.get("image_name", "Unknown")
            model = result.get("model", "Unknown")
            
            st.subheader(f"🖼️ {image_name} - {model}")
            
            extraction = result.get("extraction_result", {})
            
            if extraction.get("success"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Display structured data
                    app.display_dashboard_data(extraction["data"], f"Extracted from {image_name}")
                
                with col2:
                    # Display quality assessment
                    quality = result.get("quality_assessment", {})
                    if quality.get("success"):
                        app.display_quality_assessment(quality["assessment"])
                    else:
                        st.warning("⚠️ Quality assessment failed")
                        if quality.get("error"):
                            st.error(quality["error"])
                
                # Raw JSON (collapsible)
                with st.expander("🔍 Raw JSON Data"):
                    st.json(extraction["data"])
                
                # Usage stats
                usage = extraction.get("usage", {})
                if usage:
                    st.caption(f"Tokens: {usage.get('total_tokens', 'N/A')} | "
                              f"Prompt: {usage.get('prompt_tokens', 'N/A')} | "
                              f"Completion: {usage.get('completion_tokens', 'N/A')}")
            
            else:
                st.error(f"❌ Extraction failed: {extraction.get('error', 'Unknown error')}")
            
            st.divider()
        
        # Export results
        st.header("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON export
            json_str = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="📄 Download Complete Results (JSON)",
                data=json_str,
                file_name=f"structured_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Structured data only
            structured_only = []
            for result in results.get("results", []):
                if result.get("extraction_result", {}).get("success"):
                    structured_only.append({
                        "image": result.get("image_name"),
                        "model": result.get("model"),
                        "data": result["extraction_result"]["data"]
                    })
            
            structured_json = json.dumps(structured_only, indent=2, default=str)
            st.download_button(
                label="📊 Download Structured Data Only (JSON)",
                data=structured_json,
                file_name=f"dashboard_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    else:
        # Instructions
        st.info("👆 Configure models and click 'Run Structured Benchmark' to start")
        
        st.subheader("💡 How It Works")
        st.markdown("""
        1. **Auto-Discovery**: Automatically finds all images in `data/` folder
        2. **Structured Extraction**: Uses OpenRouter + JSON Schema for precise data extraction
        3. **Quality Assessment**: LLM evaluates extraction completeness and accuracy
        4. **Multi-Model Testing**: Compare different VLM models on the same images
        5. **Rich Visualization**: Interactive charts and quality metrics
        
        ### 🎯 What Gets Extracted:
        - **Charts**: All data points, labels, chart types
        - **Metrics**: KPIs, values, units, trends  
        - **Time Series**: Dates, values, trends over time
        - **Text Content**: Titles, labels, annotations
        - **Quality Scores**: Completeness, accuracy, structure ratings
        
        ### 🔧 Setup:
        ```bash
        # 1. Add your OpenRouter API key to .env
        cp .env.example .env
        # Edit .env: OPENROUTER_API_KEY=your_key_here
        
        # 2. Add dashboard images to data/ folder
        # 3. Run benchmark!
        ```
        """)

if __name__ == "__main__":
    main()