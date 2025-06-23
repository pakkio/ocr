"""
OCR Tester - Standalone Class for Traditional OCR Comparison
===========================================================

Pure Python class for testing and comparing traditional OCR engines:
- EasyOCR (Deep Learning based)
- PaddleOCR (AI Advanced) 
- Tesseract (Traditional)

Usage:
    tester = OCRTester()
    results = tester.compare_all_ocr(image)
    
Note: UI functionality is available in gradio_main.py
"""

import easyocr
import cv2
import numpy as np
from PIL import Image
import time
import pandas as pd
import io
import os
from typing import Dict, Any, List, Optional

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("Warning: PaddleOCR not available - install with: poetry add paddleocr")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: Tesseract not available - install with: poetry add pytesseract")

class OCRTester:
    """Standalone class for testing traditional OCR engines"""
    
    def __init__(self):
        self.easyocr_reader = None
        self.paddle_ocr = None
        
    def get_easyocr_reader(self, languages=['en', 'it']):
        """Initialize EasyOCR reader with caching"""
        if self.easyocr_reader is None:
            self.easyocr_reader = easyocr.Reader(languages, gpu=False)
        return self.easyocr_reader
    
    def get_paddle_ocr(self):
        """Initialize PaddleOCR with caching"""
        if not PADDLE_AVAILABLE:
            return None
        if self.paddle_ocr is None:
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        return self.paddle_ocr

    def test_easyocr(self, image: Image.Image) -> Dict[str, Any]:
        """Test EasyOCR on image"""
        try:
            start_time = time.time()
            
            # Convert PIL to numpy array
            img_array = np.array(image.convert('RGB'))
            
            # Run EasyOCR
            reader = self.get_easyocr_reader()
            results = reader.readtext(img_array)
            
            # Extract text and confidence
            texts = []
            confidences = []
            for (bbox, text, confidence) in results:
                texts.append(text)
                confidences.append(confidence)
            
            full_text = ' '.join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            end_time = time.time()
            
            return {
                'success': True,
                'text': full_text,
                'confidence': avg_confidence,
                'time': end_time - start_time,
                'word_count': len(texts),
                'char_count': len(full_text),
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'text': '',
                'confidence': 0.0,
                'time': 0.0,
                'word_count': 0,
                'char_count': 0,
                'error': str(e)
            }

    def test_paddleocr(self, image: Image.Image) -> Dict[str, Any]:
        """Test PaddleOCR on image"""
        if not PADDLE_AVAILABLE:
            return {
                'success': False,
                'text': '',
                'confidence': 0.0,
                'time': 0.0,
                'word_count': 0,
                'char_count': 0,
                'error': 'PaddleOCR not available'
            }
            
        try:
            start_time = time.time()
            
            # Convert PIL to numpy array
            img_array = np.array(image.convert('RGB'))
            
            # Run PaddleOCR
            ocr = self.get_paddle_ocr()
            results = ocr.ocr(img_array, cls=True)
            
            # Extract text and confidence
            texts = []
            confidences = []
            
            if results and results[0]:
                for line in results[0]:
                    if line:
                        text = line[1][0]
                        confidence = line[1][1]
                        texts.append(text)
                        confidences.append(confidence)
            
            full_text = ' '.join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            end_time = time.time()
            
            return {
                'success': True,
                'text': full_text,
                'confidence': avg_confidence,
                'time': end_time - start_time,
                'word_count': len(texts),
                'char_count': len(full_text),
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'text': '',
                'confidence': 0.0,
                'time': 0.0,
                'word_count': 0,
                'char_count': 0,
                'error': str(e)
            }

    def test_tesseract(self, image: Image.Image) -> Dict[str, Any]:
        """Test Tesseract OCR on image"""
        if not TESSERACT_AVAILABLE:
            return {
                'success': False,
                'text': '',
                'confidence': 0.0,
                'time': 0.0,
                'word_count': 0,
                'char_count': 0,
                'error': 'Tesseract not available'
            }
            
        try:
            start_time = time.time()
            
            # Run Tesseract
            text = pytesseract.image_to_string(image)
            
            # Tesseract doesn't provide confidence easily, so we estimate
            words = text.split()
            
            end_time = time.time()
            
            return {
                'success': True,
                'text': text.strip(),
                'confidence': 0.8 if len(words) > 0 else 0.0,  # Rough estimate
                'time': end_time - start_time,
                'word_count': len(words),
                'char_count': len(text.strip()),
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'text': '',
                'confidence': 0.0,
                'time': 0.0,
                'word_count': 0,
                'char_count': 0,
                'error': str(e)
            }

    def compare_all_ocr(self, image: Image.Image, engines: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Run comparison across all available OCR engines
        
        Args:
            image: PIL Image to process
            engines: List of engines to test ['easyocr', 'paddleocr', 'tesseract']
                    If None, tests all available engines
        
        Returns:
            Dictionary with results for each engine
        """
        if engines is None:
            engines = ['easyocr', 'paddleocr', 'tesseract']
            
        results = {}
        
        if 'easyocr' in engines:
            results['EasyOCR'] = self.test_easyocr(image)
            
        if 'paddleocr' in engines and PADDLE_AVAILABLE:
            results['PaddleOCR'] = self.test_paddleocr(image)
            
        if 'tesseract' in engines and TESSERACT_AVAILABLE:
            results['Tesseract'] = self.test_tesseract(image)
            
        return results

    def get_comparison_dataframe(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis"""
        comparison_data = []
        
        for engine_name, result in results.items():
            comparison_data.append({
                'Engine': engine_name,
                'Success': '✅' if result['success'] else '❌',
                'Time (s)': f"{result['time']:.3f}",
                'Confidence': f"{result['confidence']:.2f}" if result['success'] else 'N/A',
                'Words': result['word_count'],
                'Characters': result['char_count'],
                'Error': result['error'] if result['error'] else 'None'
            })
            
        return pd.DataFrame(comparison_data)

    def get_best_result(self, results: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get the best OCR result based on confidence and text length"""
        successful_results = {k: v for k, v in results.items() if v['success']}
        
        if not successful_results:
            return None
            
        # Score based on confidence * text_length
        best_engine = max(successful_results.keys(), 
                         key=lambda k: successful_results[k]['confidence'] * successful_results[k]['char_count'])
        
        return {
            'engine': best_engine,
            **successful_results[best_engine]
        }

def main():
    """Example usage of OCRTester"""
    print("🔍 OCR Tester - Pure Python Traditional OCR Comparison")
    print("=" * 60)
    print("This is a standalone class for OCR testing.")
    print("For interactive UI, use: python gradio_main.py")
    print("=" * 60)
    
    # Example usage
    tester = OCRTester()
    
    print(f"Available engines:")
    print(f"  - EasyOCR: ✅ Available")
    print(f"  - PaddleOCR: {'✅ Available' if PADDLE_AVAILABLE else '❌ Not installed'}")
    print(f"  - Tesseract: {'✅ Available' if TESSERACT_AVAILABLE else '❌ Not installed'}")
    print()
    print("To test with an image:")
    print("  from PIL import Image")
    print("  image = Image.open('your_image.jpg')")
    print("  results = tester.compare_all_ocr(image)")
    print("  df = tester.get_comparison_dataframe(results)")
    print("  print(df)")

if __name__ == "__main__":
    main()