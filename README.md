# 🤖 Ultimate Chat

A modern, multi-provider AI chat interface built with Flask. Supports OpenAI, Anthropic Claude, Google Gemini, xAI Grok, and local Ollama models.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **Multi-Provider Support**: Switch between AI providers on-the-fly
  - 🟢 OpenAI (GPT-4.1, GPT-4o, O3, O1)
  - 🟤 Anthropic (Claude Opus 4, Claude Sonnet 4, Claude 3.5)
  - 🔵 Google (Gemini 2.5 Pro, Gemini 2.0 Flash)
  - ⚫ xAI (Grok 3, Grok 2)
  - 🦙 Ollama (Llama 3.3, Qwen 2.5, DeepSeek, Mistral)
- **Beautiful Dark UI**: Modern T3-style interface with gradient accents
- **Streaming Responses**: Real-time token streaming for all providers
- **Conversation Persistence**: SQLite database for chat history
- **Document Generation**: Create PDF, Word, and PowerPoint files
- **File Uploads**: Drag & drop with image previews
- **PWA Support**: Install as a mobile/desktop app
- **API Key Management**: Configure keys through the UI
- **Connection Testing**: Verify API keys work before chatting

## 🚀 Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/LA-Rich/UltimateChat.git
cd UltimateChat

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Run the app
python app.py
```

Open http://localhost:5001 in your browser.

### Configure API Keys

1. Click the ⚙️ (Settings) button in the header
2. Enter your API keys for each provider you want to use
3. Click "Test Connection" to verify each key works
4. Select your preferred model from the dropdown

## ☁️ Deploy to Render

### One-Click Deploy

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/LA-Rich/UltimateChat)

### Manual Deployment

1. Fork this repository
2. Create a new Web Service on [Render](https://render.com)
3. Connect your GitHub repository
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120`
5. Add a **Disk** (optional, for persistent storage):
   - Mount Path: `/data`
   - Size: 1 GB
6. Add environment variables:
   - `RENDER=true`
   - `PYTHON_VERSION=3.11.4`
7. Deploy!

### Environment Variables (Render)

| Variable | Description | Required |
|----------|-------------|----------|
| `RENDER` | Set to `true` for production mode | Yes |
| `PYTHON_VERSION` | Python version (3.11.4) | Recommended |

> **Note**: API keys are configured through the UI and stored in the database, not as environment variables.

## 📁 Project Structure

```
ultimate_chat/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── .env.example           # Example environment config
├── render.yaml            # Render blueprint
├── Procfile               # Process file for deployment
├── runtime.txt            # Python version
├── static/
│   ├── manifest.json      # PWA manifest
│   └── service-worker.js  # PWA service worker
└── workspace/
    ├── uploads/           # User uploads
    ├── gallery/           # Generated images
    ├── videos/            # Generated videos
    └── documents/         # Generated documents
```

## 🔧 Configuration

### Adding Custom Models

1. Open Settings (⚙️)
2. Go to "Custom Models" tab
3. Select a provider and enter the model ID
4. Click "Add Model"

### Local Ollama Setup

```bash
# Install Ollama (https://ollama.ai)
# Pull a model
ollama pull llama3.2

# The app will auto-detect models at http://localhost:11434
```

## 📝 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main chat interface |
| `/api/providers` | GET | List all providers and models |
| `/api/settings` | GET | Get current settings |
| `/api/settings/apikey` | POST | Save API key |
| `/api/settings/model` | POST | Set current model |
| `/api/test/<provider>` | POST | Test provider connection |
| `/api/conversations` | GET | List conversations |
| `/api/chat/stream` | POST | Stream chat response |

## 🛡️ Security Notes

- API keys are stored in a local SQLite database
- Keys are never logged or exposed in the UI
- For production, use environment variables or a secrets manager
- The `.env` file is gitignored by default

## 📄 License

MIT License - feel free to use this project for any purpose.

## 🙏 Acknowledgments

- [Flask](https://flask.palletsprojects.com/) - Web framework
- [Marked.js](https://marked.js.org/) - Markdown rendering
- [Highlight.js](https://highlightjs.org/) - Code syntax highlighting
- All the amazing AI providers making this possible!

