import time
import asyncio
from typing import Optional
from PIL import Image
import numpy as np
import cv2

from .base import BaseOCRProvider, OCRResult

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

class TraditionalOCRProvider(BaseOCRProvider):
    """Traditional OCR provider for EasyOCR, PaddleOCR, and Tesseract"""
    
    def __init__(self, config):
        super().__init__(config)
        self.provider_name = "traditional"
        self._easyocr_reader = None
        self._paddle_ocr = None
    
    def _get_easyocr_reader(self, languages=['en', 'it']):
        """Lazy initialization of EasyOCR reader"""
        if not EASYOCR_AVAILABLE:
            return None
        if self._easyocr_reader is None:
            self._easyocr_reader = easyocr.Reader(languages)
        return self._easyocr_reader
    
    def _get_paddle_ocr(self):
        """Lazy initialization of PaddleOCR"""
        if not PADDLEOCR_AVAILABLE:
            return None
        if self._paddle_ocr is None:
            self._paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
        return self._paddle_ocr
    
    async def extract_text_easyocr(self, image: Image.Image) -> OCRResult:
        """Extract text using EasyOCR"""
        start_time = time.time()
        
        if not EASYOCR_AVAILABLE:
            return OCRResult(
                text="",
                execution_time=0,
                error="EasyOCR not available",
                provider="traditional",
                model="easyocr"
            )
        
        try:
            reader = self._get_easyocr_reader()
            
            # Convert PIL to numpy array
            image_array = np.array(image)
            
            # Run OCR in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                reader.readtext, 
                image_array
            )
            
            # Extract text and confidence
            text_lines = [result[1] for result in results]
            confidences = [result[2] for result in results]
            
            text = '\n'.join(text_lines)
            avg_confidence = np.mean(confidences) if confidences else 0
            
            execution_time = time.time() - start_time
            
            return OCRResult(
                text=text,
                execution_time=execution_time,
                confidence=avg_confidence,
                provider="traditional",
                model="easyocr",
                metadata={
                    'detection_count': len(results),
                    'raw_results': results
                }
            )
            
        except Exception as e:
            return OCRResult(
                text="",
                execution_time=time.time() - start_time,
                error=str(e),
                provider="traditional",
                model="easyocr"
            )
    
    async def extract_text_paddleocr(self, image: Image.Image) -> OCRResult:
        """Extract text using PaddleOCR"""
        start_time = time.time()
        
        if not PADDLEOCR_AVAILABLE:
            return OCRResult(
                text="",
                execution_time=0,
                error="PaddleOCR not available",
                provider="traditional",
                model="paddleocr"
            )
        
        try:
            ocr = self._get_paddle_ocr()
            
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Run OCR in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                ocr.ocr,
                cv_image,
                True  # cls parameter
            )
            
            # Extract text and confidence
            text_lines = []
            confidences = []
            
            if results and results[0]:
                for line in results[0]:
                    if line:
                        text_lines.append(line[1][0])
                        confidences.append(line[1][1])
            
            text = '\n'.join(text_lines)
            avg_confidence = np.mean(confidences) if confidences else 0
            
            execution_time = time.time() - start_time
            
            return OCRResult(
                text=text,
                execution_time=execution_time,
                confidence=avg_confidence,
                provider="traditional",
                model="paddleocr",
                metadata={
                    'detection_count': len(text_lines),
                    'raw_results': results
                }
            )
            
        except Exception as e:
            return OCRResult(
                text="",
                execution_time=time.time() - start_time,
                error=str(e),
                provider="traditional",
                model="paddleocr"
            )
    
    async def extract_text_tesseract(self, image: Image.Image) -> OCRResult:
        """Extract text using Tesseract"""
        start_time = time.time()
        
        if not TESSERACT_AVAILABLE:
            return OCRResult(
                text="",
                execution_time=0,
                error="Tesseract not available",
                provider="traditional",
                model="tesseract"
            )
        
        try:
            # Run Tesseract in thread pool
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None,
                pytesseract.image_to_string,
                image,
                'eng+ita'
            )
            
            execution_time = time.time() - start_time
            
            return OCRResult(
                text=text.strip(),
                execution_time=execution_time,
                confidence=0.8,  # Tesseract doesn't provide easy confidence access
                provider="traditional",
                model="tesseract"
            )
            
        except Exception as e:
            return OCRResult(
                text="",
                execution_time=time.time() - start_time,
                error=str(e),
                provider="traditional",
                model="tesseract"
            )
    
    async def extract_text(
        self, 
        image: Image.Image, 
        model: str = "easyocr",
        prompt: Optional[str] = None
    ) -> OCRResult:
        """Extract text using specified traditional OCR model"""
        
        if model == "easyocr":
            return await self.extract_text_easyocr(image)
        elif model == "paddleocr":
            return await self.extract_text_paddleocr(image)
        elif model == "tesseract":
            return await self.extract_text_tesseract(image)
        else:
            return OCRResult(
                text="",
                execution_time=0,
                error=f"Unknown traditional OCR model: {model}",
                provider="traditional",
                model=model
            )
    
    async def batch_extract_text(
        self, 
        images: list[Image.Image], 
        model: str = "easyocr"
    ) -> list[OCRResult]:
        """Extract text from multiple images"""
        tasks = [
            self.extract_text(image, model) 
            for image in images
        ]
        return await asyncio.gather(*tasks)
    
    @staticmethod
    def get_available_models() -> dict:
        """Get available traditional OCR models"""
        return {
            "easyocr": {
                "name": "EasyOCR",
                "available": EASYOCR_AVAILABLE,
                "description": "Deep learning OCR with 80+ language support"
            },
            "paddleocr": {
                "name": "PaddleOCR", 
                "available": PADDLEOCR_AVAILABLE,
                "description": "Advanced OCR with excellent accuracy"
            },
            "tesseract": {
                "name": "Tesseract",
                "available": TESSERACT_AVAILABLE,
                "description": "Traditional OCR engine"
            }
        }