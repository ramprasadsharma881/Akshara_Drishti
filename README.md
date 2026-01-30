# AksharaDrishti - Indic OCR for Vedantic Literature

> **"Akshara"** (अक्षर) = Letter/Character | **"Drishti"** (दृष्टि) = Vision  
> *The Vision that Sees Every Letter*

## 📖 What is AksharaDrishti?

**AksharaDrishti** is an advanced OCR (Optical Character Recognition) application specifically engineered for **Kannada, Telugu, and Sanskrit** texts. Unlike generic OCR tools, it is purpose-built for **Vedantic, philosophical, and classical Indian literature** — the kind of texts that existing OCR solutions struggle with.

It combines:
- **Tesseract OCR** with specialized preprocessing
- **AI-powered spell correction** using LLMs (Gemini/OpenAI)
- **Intelligent text formatting** for shlokas, verses, and headings
- **Linguistic repair** for Sandhi and Danda normalization

## 🎯 Who Is This For?

| User Type | Use Case |
|-----------|----------|
| **Sanskrit Scholars** | Digitize ancient manuscripts and commentaries |
| **Vedantic Institutions** | Convert printed books to searchable digital format |
| **Publishers** | Create editable Word documents from scanned religious texts |
| **Researchers** | Extract text from old Kannada/Telugu philosophical works |
| **Libraries & Archives** | Preserve heritage literature in digital form |
| **Students** | Create study materials from scanned textbooks |

**Perfect for:** Upanishads, Bhashyas (commentaries), Stotras, Gitas, and any classical Indic text with mixed Sanskrit quotes.

## ✨ What Makes AksharaDrishti Different?

### vs. Google Lens / Generic OCR
| Feature | Generic OCR | AksharaDrishti |
|---------|-------------|----------------|
| Shloka/Verse detection | ❌ No | ✅ Auto-bold with proper spacing |
| Danda (।॥) handling | ❌ Often corrupted | ✅ Normalized correctly |
| Mixed-script text | ❌ Poor accuracy | ✅ Kannada+Sanskrit, Telugu+Sanskrit |
| Conjunct consonants | ❌ Frequently broken | ✅ Enhanced preprocessing |
| AI spell correction | ❌ No | ✅ Gemini/OpenAI integration |
| Word document output | ❌ Plain text only | ✅ Formatted .docx with styles |

### vs. ABBYY / Commercial OCR
| Feature | Commercial OCR | AksharaDrishti |
|---------|----------------|----------------|
| Indian language support | ⚠️ Limited | ✅ Native Kannada/Telugu/Sanskrit |
| Vedantic text formatting | ❌ No | ✅ Heading, verse, shloka detection |
| Cost | 💰 Expensive licenses | 🆓 Free & Open Source |
| AI correction for Indic | ❌ No | ✅ Context-aware spell fixing |

### 🔑 Unique Features

1. **Vedantic-Aware Formatting**
   - Auto-detects chapter names (अध्याय, ಅಧ್ಯಾಯ, అధ్యాయ)
   - Identifies verse numbers (॥ 12 ॥) and Om invocations
   - Bolds shlokas while keeping prose normal

2. **Linguistic Repair Layer**
   - Fixes Sandhi breaks (hyphenated word joins)
   - Normalizes Dandas (I/l/1 → ।॥)
   - Removes OCR artifacts (* noise, garbage characters)

3. **AI Spell Correction**
   - Uses Gemini or OpenAI to fix OCR errors
   - Preserves technical Sanskrit terms and mantras
   - Conservative correction — only fixes obvious mistakes

4. **Quality Reporting**
   - Confidence scoring per page
   - Low-confidence word tracking
   - Summary report at end of document

## 🧠 Algorithmic Approaches for Maximum Accuracy

AksharaDrishti achieves **Cloud Vision-level accuracy** through a multi-stage pipeline:

### 1. Image Preprocessing — "Clean & Pop" Pipeline

| Stage | Algorithm | Purpose |
|-------|-----------|---------|
| **Denoising** | Non-local Means Denoising (`cv2.fastNlMeansDenoising`) | Removes paper noise/aging artifacts without destroying thin Indic strokes |
| **Contrast Enhancement** | CLAHE (Contrast Limited Adaptive Histogram Equalization) | Critical for text near book bindings where shadows darken the page |
| **Binarization** | Adaptive Gaussian Thresholding | Creates razor-sharp black/white image for Tesseract's LSTM engine |
| **Context Padding** | 30px white border addition | Helps recognize Vattulu (subscript consonants) and edge characters |
| **Header/Footer Removal** | 8% top, 6% bottom cropping | Removes repeated book titles and page numbers |

### 2. OCR Engine Configuration

```
Tesseract Configuration:
├── OEM 1: LSTM-only mode (neural network, best for Indic scripts)
├── PSM 6: Uniform block of text (optimal for printed books)
└── preserve_interword_spaces=1: Maintains original word spacing
```

### 3. Linguistic Repair Layer

| Repair Type | Problem | Solution |
|-------------|---------|----------|
| **Sandhi Joining** | Line-break hyphenation (`Veda-\nnta`) | Regex-based word joining across lines |
| **Danda Normalization** | OCR reads `।` as `I`, `l`, `1` | Pattern matching to restore `।` and `॥` |
| **Virama Handling** | Broken conjuncts at line breaks | Joins consonant+virama with next consonant |
| **Matra Orphaning** | Vowel signs separated from consonants | Reattaches dependent vowels to base characters |
| **Artifact Removal** | Asterisks, garbage characters | Regex cleanup of OCR noise |

### 4. Text Classification (Conservative Pattern Matching)

```
Classification Hierarchy:
├── Verse Detection
│   ├── Verse number patterns: ॥ 12 ॥
│   ├── Om invocations: ॐ, ಓಂ, ఓం
│   └── Short lines (<120 chars) ending with ॥
├── Heading Detection
│   ├── Chapter keywords: अध्याय, ಅಧ್ಯಾಯ, అధ్యాయ
│   ├── Numbered sections (if <50 chars)
│   └── Uvacha patterns (speaker introductions)
└── Sanskrit Detection
    ├── >70% Devanagari script
    └── <20% mixed local script (Kannada/Telugu)
```

### 5. Parallel Processing Architecture

- **ProcessPoolExecutor** for page-level parallelism
- **Thread limiting**: `cv2.setNumThreads(2)` + `OMP_NUM_THREADS=2` to prevent exhaustion
- **Fault isolation**: Failed pages don't crash the entire job
- **Dynamic worker count**: `min(8, max(2, cpu_count // 4))`

### 6. AI Post-Processing (Optional)

- **LLM Spell Correction**: Gemini/OpenAI fixes OCR errors
- **Conservative prompting**: Only fixes obvious mistakes
- **Validation layer**: Rejects corrections that change >30% of text
- **Indic script preservation**: Verifies script character ratios

## 🔑 API Keys Configuration

**No API keys are bundled with this application.** You need to provide your own.

### For AI Spell Correction (Optional but Recommended)

You can use **FREE** API keys from OpenRouter to access Gemini models:

1. **Get Free OpenRouter API Key:**
   - Go to [OpenRouter.ai](https://openrouter.ai/)
   - Sign up for free
   - Get your API key
   - Use free Gemini models (gemini-2.0-flash, gemini-pro, etc.)

2. **Set Environment Variables:**
   ```bash
   # Option 1: OpenAI (paid)
   set OPENAI_API_KEY=your_openai_key_here
   
   # Option 2: Use OpenRouter with free Gemini models
   # Modify openai_spell_check.py to use OpenRouter endpoint
   # OPENAI_API_URL = "https://openrouter.ai/api/v1/chat/completions"
   
   # Option 3: Direct Gemini via Helicone
   set HELICONE_API_KEY=your_helicone_key_here
   
   # Enable/disable spell check
   set SPELL_CHECK_ENABLED=true
   ```

> **💡 Tip:** OpenRouter provides FREE access to Gemini models — perfect for testing and personal use!

## 🏠 Local Installation

### Prerequisites
- Python 3.10+
- Tesseract OCR with Indic language packs
- Poppler for PDF processing

### Windows Setup
```bash
# 1. Install Tesseract OCR
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Install with Kannada, Telugu, Sanskrit language packs

# 2. Install Poppler
# Download from: https://github.com/osber/poppler-windows/releases

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Run the application
python app.py
```

### Linux/Mac Setup
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-kan tesseract-ocr-tel tesseract-ocr-san poppler-utils

# Install Python dependencies
pip install -r requirements.txt

# Run
python app.py
```

Open browser at `http://localhost:8000`

## 📝 How to Use

1. **Upload** a PDF file (scanned book/document)
2. **Select language** (Kannada+Sanskrit, Telugu+Sanskrit, etc.)
3. **Set DPI** (300-400 recommended for old prints)
4. **Click "Start OCR Processing"**
5. **Watch progress** with real-time updates
6. **Download** the formatted Word document

## 🛠️ Technical Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI (Python) |
| OCR Engine | Tesseract with LSTM |
| Preprocessing | OpenCV (CLAHE, Denoising, Adaptive Threshold) |
| AI Correction | OpenAI GPT / Gemini |
| PDF Processing | pdf2image + Poppler |
| Document Output | python-docx |
| Parallel Processing | ProcessPoolExecutor |

## � Project Structure

```
AksharaDrishti/
├── app.py                  # FastAPI web server & API endpoints
├── akshara_desktop.py      # Desktop GUI application (Tkinter)
├── ocr_utils.py            # Core OCR, preprocessing & formatting
├── openai_spell_check.py   # AI spell correction (OpenAI/OpenRouter)
├── gemini_spell_check.py   # AI spell correction (Gemini/Helicone)
├── requirements.txt        # Python dependencies
├── static/                 # Web UI files
└── README.md               # This file
```

## 🎯 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/upload` | POST | Upload PDF and start OCR |
| `/api/status/{job_id}` | GET | Get processing status |
| `/api/download/{job_id}` | GET | Download result |
| `/api/health` | GET | Health check |

## � Tips for Best Results

- **Use 300-400 DPI** for scanned documents
- **Clean scans** produce better results than photographs
- **Select correct language** — mixed scripts need correct combination
- **Enable AI spell check** for best accuracy (requires API key)

## 🔒 Privacy & Security

- **Files are deleted** after processing completes
- **No data stored** — job data is in-memory only
- **API keys are environment variables** — never hardcoded
- **Local processing** — your documents don't leave your machine (unless using cloud AI)

## 📞 Support

For issues, feature requests, or questions:
- Create an issue on GitHub
- Check the `explanation.txt` file for detailed technical documentation

---

**Made with ❤️ for preserving Vedantic literature in the digital age**

*AksharaDrishti — Where ancient wisdom meets modern technology*
