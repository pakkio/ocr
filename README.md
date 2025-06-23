# 🔬 Advanced OCR Benchmark Suite

Piattaforma completa per testare e confrontare **OCR tradizionali** vs **Vision Language Models** moderni con AI, featuring **structured JSON extraction** e **quality assessment** automatizzato.

## Modelli Supportati

### 🤖 Vision Language Models (via OpenRouter)
- **GPT-4 Vision Preview & GPT-4o** - OpenAI
- **Claude 3.5 Sonnet & Haiku** - Anthropic  
- **Gemini Pro & Flash 1.5** - Google
- **Mistral Pixtral 12B** - Mistral AI
- **Qwen2-VL 72B & 7B** - Alibaba

### 🔧 Traditional OCR
- **EasyOCR**: Deep Learning, 80+ lingue
- **PaddleOCR**: AI avanzato, ottimo per documenti complessi
- **Tesseract**: Engine tradizionale di riferimento

## Installazione

```bash
# Installa Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Installa dipendenze
poetry install

# Dipendenze sistema (Ubuntu/Debian):
sudo apt-get install libgl1-mesa-glx libglib2.0-0 tesseract-ocr tesseract-ocr-ita

# Configura API keys
cp .env.example .env
# Modifica .env con le tue API keys
```

## Utilizzo

```bash
# 🚀 GRADIO APP (RACCOMANDATO) - Modern UI con 3 modalità
poetry run python gradio_app.py

# Legacy Streamlit apps (opzionali)
poetry run streamlit run ocr_tester.py              # Traditional only
poetry run streamlit run advanced_ocr_app.py        # Hybrid mode  
poetry run streamlit run structured_benchmark.py    # Structured only
```

## Features Avanzate

### 🚀 Benchmark Completo
- Test batch su multiple immagini
- Confronto simultaneo di tutti i modelli
- Metriche dettagliate: tempo, accuratezza, costo
- Analisi statistiche e visualizzazioni

### 🔬 Structured JSON Extraction
- **Auto-discovery** immagini in `data/` 
- **JSON Schema** validazione con Pydantic
- **Quality Assessment** via LLM per valutare accuratezza
- **Structured Output** per dashboard e grafici complessi
- **Multi-Model Comparison** su dati strutturati

### 🎨 Modern Gradio Interface
- **Mobile-responsive** design automatico
- **Real-time progress** tracking nativo
- **3-tab layout**: Structured, Traditional, Batch
- **Modern theme** con UX ottimizzata
- **Easy sharing** per demo pubblici

### 🏗️ Architettura Enterprise
- **Factory Pattern** per providers OCR
- **Dependency Injection** per configurazioni
- **Async processing** per performance
- **OpenRouter integration** per accesso unificato ai VLM
- **Error handling** con retry logic robusto

## 🔑 Setup API Keys

### OpenRouter (Raccomandato)
Crea un account su [OpenRouter](https://openrouter.ai/keys) per accesso unificato:
- **Un singolo account** per GPT-4o, Claude 3.5, Gemini, Mistral, Qwen2-VL
- **Pricing trasparente** pay-per-use senza subscription
- **No vendor lock-in** - switch tra modelli istantaneamente
- **Rate limits alti** per testing e production

### Setup Velocissimo
```bash
# 1. Ottieni API key: https://openrouter.ai/keys
# 2. Configura in 10 secondi:
cp .env.example .env
echo "OPENROUTER_API_KEY=sk-or-v1-your-key-here" >> .env
# 3. Launch!
poetry run python gradio_app.py
```

## 📋 Modalità di Utilizzo

### 🚀 1. Gradio App (RACCOMANDATO)
**Modern UI** con 3 modalità integrate in un'unica interfaccia:

```bash
poetry run python gradio_app.py
```

**3 Tab principali:**
- **🔬 Structured JSON**: VLM extraction + quality assessment  
- **🔧 Traditional OCR**: EasyOCR, PaddleOCR, Tesseract comparison
- **📁 Batch Processing**: Auto-process all images in `data/`

**Vantaggi Gradio:**
- ✅ **Mobile-friendly** responsive design
- ✅ **Real-time progress** bars nativi
- ✅ **90% less code** rispetto a Streamlit
- ✅ **Modern UX** con tema Soft
- ✅ **Easy sharing** con `share=True`

### 🔧 2. Legacy Streamlit Apps (Opzionali)
Per compatibility o use case specifici:
- `ocr_tester.py` - Traditional OCR only
- `advanced_ocr_app.py` - Hybrid benchmark  
- `structured_benchmark.py` - JSON extraction only

## 🎯 Casi d'Uso

### 📊 Business Intelligence
- **Dashboard Analysis**: Estrazione dati automatica da screenshot analytics
- **Report Processing**: Conversione automatica grafici → JSON strutturato
- **KPI Monitoring**: Extraction metriche da dashboard esistenti

### 🔬 Ricerca e Sviluppo
- **Model Comparison**: Confronto sistematico VLM vs OCR tradizionali
- **Accuracy Assessment**: Valutazione qualità extraction via LLM
- **Performance Benchmarking**: Analisi velocità/accuratezza/costo

### 🏢 Enterprise Applications
- **Document Digitization**: Conversion automatica documenti complessi
- **Data Migration**: Export strutturato da sistemi legacy
- **Quality Assurance**: Validazione automatica extraction accuracy
- **Demo & Prototyping**: Interface pronta per stakeholder demos

## 📈 Dataset di Test Inclusi

Il repository include **3 dashboard complessi** nella cartella `data/`:
- **Business Dashboard**: Metriche earnings, downloads, grafici temporali
- **Analytics Dashboard**: KPI, percentuali, widget multipli  
- **Multilingual Dashboard**: Testo latino, grafici colorati, time series

Ideali per testare:
- ✅ **Numeric extraction** (valute, percentuali, grandi numeri)
- ✅ **Chart data points** (labels, valori, tipologie)
- ✅ **Complex layouts** (widget sovrapposti, colori di sfondo)
- ✅ **Text recognition** (font piccoli, watermark filtering)
- ✅ **Time series data** (date, trend, periodi)

## 🚀 Quick Start (2 minuti)

```bash
# 1. Clone & setup
git clone <repo-url>
cd ocr
poetry install

# 2. Configure API key  
cp .env.example .env
# Edit .env: OPENROUTER_API_KEY=sk-or-v1-your-key

# 3. Launch modern Gradio app
poetry run python gradio_app.py
# 🌐 Open: http://localhost:7860

# 4. Test with included dashboards
# Your 3 sample images are auto-discovered in data/ folder!
```

### 🎯 Prima volta?
1. **Tab "Structured JSON"** → Upload immagine → Select "gpt-4o" → Click "Extract"
2. **Guarda JSON** estratto + quality score 0-10
3. **Confronta** con Traditional OCR nel secondo tab
4. **Batch process** tutte le immagini nel terzo tab