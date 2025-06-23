import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image
import time
import pandas as pd
import io
import os

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    st.warning("PaddleOCR non disponibile - installa con: poetry add paddleocr")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    st.warning("Tesseract non disponibile - installa con: poetry add pytesseract")

class OCRTester:
    def __init__(self):
        self.easyocr_reader = None
        self.paddle_ocr = None
        
    @st.cache_resource
    def get_easyocr_reader(_self, languages=['en', 'it']):
        return easyocr.Reader(languages)
    
    @st.cache_resource 
    def get_paddle_ocr(_self):
        if PADDLE_AVAILABLE:
            return PaddleOCR(use_angle_cls=True, lang='en')
        return None
    
    def preprocess_image(self, image):
        """Converte immagine PIL in formato OpenCV"""
        if isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return image
    
    def test_easyocr(self, image):
        """Test con EasyOCR"""
        start_time = time.time()
        
        if self.easyocr_reader is None:
            self.easyocr_reader = self.get_easyocr_reader()
        
        # EasyOCR accetta direttamente array numpy
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
            
        results = self.easyocr_reader.readtext(image_array)
        
        execution_time = time.time() - start_time
        
        # Estrai solo il testo
        text = '\n'.join([result[1] for result in results])
        
        return {
            'text': text,
            'time': execution_time,
            'confidence': np.mean([result[2] for result in results]) if results else 0,
            'details': results
        }
    
    def test_paddleocr(self, image):
        """Test con PaddleOCR"""
        if not PADDLE_AVAILABLE:
            return {'text': 'PaddleOCR non disponibile', 'time': 0, 'confidence': 0}
            
        start_time = time.time()
        
        if self.paddle_ocr is None:
            self.paddle_ocr = self.get_paddle_ocr()
        
        # Converti in formato OpenCV
        cv_image = self.preprocess_image(image)
        
        results = self.paddle_ocr.ocr(cv_image, cls=True)
        
        execution_time = time.time() - start_time
        
        # Estrai testo e confidence
        text_lines = []
        confidences = []
        
        if results and results[0]:
            for line in results[0]:
                if line:
                    text_lines.append(line[1][0])
                    confidences.append(line[1][1])
        
        text = '\n'.join(text_lines)
        avg_confidence = np.mean(confidences) if confidences else 0
        
        return {
            'text': text,
            'time': execution_time,
            'confidence': avg_confidence,
            'details': results
        }
    
    def test_tesseract(self, image):
        """Test con Tesseract"""
        if not TESSERACT_AVAILABLE:
            return {'text': 'Tesseract non disponibile', 'time': 0, 'confidence': 0}
            
        start_time = time.time()
        
        # Tesseract lavora meglio con PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        text = pytesseract.image_to_string(image, lang='eng+ita')
        
        execution_time = time.time() - start_time
        
        return {
            'text': text.strip(),
            'time': execution_time,
            'confidence': 0.8,  # Tesseract non fornisce confidence facilmente
            'details': None
        }

def main():
    st.set_page_config(
        page_title="OCR Tester - Confronto AI vs Tradizionale",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 OCR Tester: EasyOCR vs PaddleOCR vs Tesseract")
    st.markdown("**Confronto tra OCR moderni con AI e soluzioni tradizionali**")
    
    ocr_tester = OCRTester()
    
    # Sidebar per configurazioni
    st.sidebar.header("⚙️ Configurazioni")
    
    # Upload file
    uploaded_file = st.sidebar.file_uploader(
        "Carica un'immagine",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Formati supportati: PNG, JPG, JPEG, BMP, TIFF"
    )
    
    # Selezione OCR da testare
    st.sidebar.subheader("OCR da testare:")
    test_easyocr = st.sidebar.checkbox("EasyOCR (Deep Learning)", value=True)
    test_paddleocr = st.sidebar.checkbox("PaddleOCR (AI Avanzato)", value=True)
    test_tesseract = st.sidebar.checkbox("Tesseract (Tradizionale)", value=True)
    
    if uploaded_file is not None:
        # Mostra immagine
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📷 Immagine Originale")
            st.image(image, caption="Immagine da analizzare", use_column_width=True)
            
            st.info(f"**Dimensioni:** {image.size[0]}x{image.size[1]} px")
        
        with col2:
            st.subheader("🚀 Risultati OCR")
            
            if st.button("▶️ Avvia Test OCR", type="primary"):
                results = {}
                
                # Test EasyOCR
                if test_easyocr:
                    with st.spinner("🔄 Testando EasyOCR..."):
                        results['EasyOCR'] = ocr_tester.test_easyocr(image)
                
                # Test PaddleOCR  
                if test_paddleocr:
                    with st.spinner("🔄 Testando PaddleOCR..."):
                        results['PaddleOCR'] = ocr_tester.test_paddleocr(image)
                
                # Test Tesseract
                if test_tesseract:
                    with st.spinner("🔄 Testando Tesseract..."):
                        results['Tesseract'] = ocr_tester.test_tesseract(image)
                
                # Mostra risultati
                if results:
                    st.success("✅ Test completati!")
                    
                    # Tabella comparativa
                    st.subheader("📊 Confronto Prestazioni")
                    
                    comparison_data = []
                    for ocr_name, result in results.items():
                        comparison_data.append({
                            'OCR': ocr_name,
                            'Tempo (s)': f"{result['time']:.3f}",
                            'Confidence': f"{result['confidence']:.2%}",
                            'Caratteri': len(result['text'])
                        })
                    
                    df = pd.DataFrame(comparison_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Risultati dettagliati
                    st.subheader("📝 Testi Estratti")
                    
                    for ocr_name, result in results.items():
                        with st.expander(f"🔍 {ocr_name} (Tempo: {result['time']:.3f}s)"):
                            if result['text']:
                                st.text_area(
                                    f"Testo estratto da {ocr_name}:",
                                    result['text'],
                                    height=150,
                                    key=f"text_{ocr_name}"
                                )
                            else:
                                st.warning("Nessun testo rilevato")
    
    else:
        st.info("👆 Carica un'immagine dalla sidebar per iniziare il test")
        
        # Esempio di utilizzo
        st.subheader("💡 Come usare l'app")
        st.markdown("""
        1. **Carica un'immagine** dalla sidebar (formati: PNG, JPG, JPEG, BMP, TIFF)
        2. **Seleziona gli OCR** che vuoi testare
        3. **Clicca su "Avvia Test OCR"** per confrontare le prestazioni
        4. **Analizza i risultati** nella tabella comparativa e nei dettagli
        
        ### 🎯 OCR Disponibili:
        - **EasyOCR**: Veloce e semplice, buono per uso generale
        - **PaddleOCR**: Più accurato, migliore per documenti complessi  
        - **Tesseract**: Il veterano, riferimento storico per confronti
        """)

if __name__ == "__main__":
    main()