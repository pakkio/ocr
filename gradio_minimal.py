import gradio as gr
import os

def create_minimal_app():
    """Create minimal Gradio app without Pydantic dependencies"""
    
    with gr.Blocks(
        title="🔬 OCR Benchmark Suite - Minimal",
        theme=gr.themes.Soft()
    ) as app:
        
        gr.Markdown("""
        # 🔬 OCR Benchmark Suite - Minimal Version
        
        This is a minimal version to test Gradio compatibility without Pydantic schema conflicts.
        """)
        
        with gr.Tab("🔧 Simple Test"):
            with gr.Row():
                with gr.Column():
                    test_input = gr.Textbox(
                        label="Test Input",
                        placeholder="Enter some text..."
                    )
                    test_btn = gr.Button("Process", variant="primary")
                
                with gr.Column():
                    test_output = gr.Textbox(
                        label="Output",
                        interactive=False
                    )
            
            def process_text(text):
                return f"Processed: {text}"
            
            test_btn.click(
                fn=process_text,
                inputs=[test_input],
                outputs=[test_output]
            )
        
        gr.Markdown("""
        ### ✅ Status
        If you see this interface, Gradio is working correctly without schema conflicts.
        """)
    
    return app

if __name__ == "__main__":
    app = create_minimal_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        show_error=True,
        debug=True
    )