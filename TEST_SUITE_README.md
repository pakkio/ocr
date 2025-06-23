# 🧪 Comprehensive OCR Test Suite

A persistent test suite that evaluates all OCR modes, models, and approaches explored in this project.

## 🎯 Purpose

This test suite provides automated validation of:
- **VLM Models**: All OpenAI, Anthropic, and Google vision models
- **Traditional OCR**: EasyOCR, PaddleOCR, Tesseract
- **Multiple Approaches**: Structured JSON extraction vs simple text extraction
- **Error Handling**: Fallback mechanisms and graceful error handling
- **Performance**: Execution time and success rates across all providers

## 🚀 Quick Start

### Run a Quick Test (Recommended)
```bash
python run_tests.py --mode quick
```
Tests a subset of models with one image (~2-3 minutes).

### Run Full Test Suite
```bash
python run_tests.py --mode full
```
Tests all models with all images (~10-15 minutes).

### Test Specific Provider
```bash
python run_tests.py --mode provider --provider openai
python run_tests.py --mode provider --provider anthropic
python run_tests.py --mode provider --provider google
python run_tests.py --mode provider --provider traditional
```

### Run Directly
```bash
python comprehensive_test_suite.py
```
Runs the full test suite directly.

## 📊 Test Categories

### 1. VLM Structured Extraction
Tests all vision language models using the structured provider with Pydantic schema validation:
- **OpenAI**: gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano
- **Anthropic**: claude-3.5-sonnet, claude-sonnet-4, claude-3.7-sonnet  
- **Google**: gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite, gemini-pro-1.5, gemini-flash-1.5

**Schema Modes**:
- `strict`: Uses OpenRouter's strict JSON schema mode
- `fallback`: Uses basic `json_object` mode when strict schema fails

### 2. VLM Gradio Mode
Tests key models using the Gradio app's simpler JSON object approach:
- Faster execution
- More forgiving schema requirements
- Better compatibility across models

### 3. Traditional OCR
Tests classic OCR engines for text extraction:
- **EasyOCR**: General purpose OCR
- **PaddleOCR**: Chinese-focused OCR with good English support
- **Tesseract**: Google's OCR engine

### 4. Error Handling
Tests robustness and fallback mechanisms:
- Invalid model names
- Corrupted image data
- API failures and timeouts
- Schema validation errors

## 📈 Output Files

### Test Results JSON
`test_results_YYYYMMDD_HHMMSS.json`
- Complete machine-readable results
- Detailed timing and error information
- Raw data for further analysis

### Human-Readable Report  
`test_report_YYYYMMDD_HHMMSS.md`
- Markdown formatted summary
- Success rates by provider
- Performance comparisons
- Schema mode effectiveness

### Log File
`test_results.log`
- Real-time test execution log
- Error messages and debugging info
- Persistent across multiple runs

## 🔧 Configuration

### Test Images
Default test images (customizable in `ComprehensiveTestSuite`):
```python
self.test_images = [
    "data/istockphoto-1303610445-1024x1024.jpg",  # Business dashboard
    "data/istockphoto-1390723387-1024x1024.jpg",  # Analytics dashboard  
    "data/istockphoto-1472103438-1024x1024.jpg"   # Multilingual dashboard
]
```

### Model Lists
Organized by provider (easily customizable):
```python
self.vlm_models = {
    "OpenAI": ["gpt-4o", "openai/gpt-4o-mini", ...],
    "Anthropic": ["anthropic/claude-3.5-sonnet", ...],
    "Google": ["google/gemini-2.5-pro", ...]
}
```

## 📋 Understanding Results

### Success Rate Interpretation
- **>90%**: Excellent - All major models working correctly
- **80-90%**: Good - Most models working, some expected failures
- **70-80%**: Fair - Several models having issues, investigate
- **<70%**: Poor - Major compatibility problems, needs attention

### Common Issues
1. **Schema Validation Errors**: Models returning data in unexpected format
2. **API Rate Limits**: Too many requests too quickly
3. **Model Unavailability**: Temporary outages or deprecated models
4. **Authentication**: Missing or invalid API keys

### Performance Metrics
- **Execution Time**: How long each model takes
- **Data Extraction**: Whether actual data was extracted (not just success/failure)
- **Schema Compliance**: Which models follow strict vs fallback modes

## 🔍 Troubleshooting

### Environment Setup
Ensure you have:
```bash
# Required environment variable
export OPENROUTER_API_KEY="your_key_here"

# Required dependencies
pip install -r requirements.txt
```

### Common Fixes
1. **"Module not found"**: Run from project root directory
2. **"API key not found"**: Check `.env` file or environment variables
3. **"Image not found"**: Ensure test images exist in `data/` directory
4. **High failure rate**: Check network connectivity and API quotas

### Debug Mode
Add logging for more detailed output:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## 🎛️ Customization

### Add New Models
1. Update model lists in `ComprehensiveTestSuite.__init__()`
2. Add model configuration in `src/config.py`
3. Set appropriate `supports_strict_json_schema` flag

### Add New Test Images
1. Place images in `data/` directory
2. Update `self.test_images` list
3. Ensure images are complex enough for meaningful testing

### Modify Test Criteria
Customize success criteria in test methods:
- Minimum chart/metric extraction counts
- Required confidence levels
- Acceptable error types

## 🚀 Integration

### CI/CD Pipeline
```yaml
name: OCR Model Testing
on: [push, pull_request, schedule]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run OCR Tests
        run: python run_tests.py --mode quick
      - name: Upload Results
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: test_*.json
```

### Monitoring
Set up alerts for:
- Success rate drops below threshold
- New model failures
- Performance degradation

### Scheduled Testing
```bash
# Run daily at 2 AM
0 2 * * * cd /path/to/ocr && python run_tests.py --mode full
```

## 📚 Best Practices

1. **Regular Testing**: Run quick tests after code changes
2. **Full Testing**: Run complete suite weekly or before releases
3. **Trend Analysis**: Monitor success rates over time
4. **Model Updates**: Update model lists when new versions are released
5. **Performance Baseline**: Track execution time trends

## 🤝 Contributing

To add new test scenarios:
1. Create new test methods in `ComprehensiveTestSuite`
2. Add corresponding result tracking
3. Update documentation
4. Test with existing models to ensure no regressions

---

**Happy Testing!** 🎯