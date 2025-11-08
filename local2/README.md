<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# GPU Kernel Translator

A powerful web application that translates GPU kernels between CUDA, Triton, and Mojo using Ollama AI, with remote server verification via SSH.

## Features

- 🔄 **AI-Powered Translation**: Translate GPU kernels between CUDA, Triton, and Mojo using Ollama's `gemma3n:latest` model
- 🔍 **Remote Verification**: Verify translated code on a remote server via SSH
- 🎯 **Feedback Loop**: Provide compiler errors or feedback to regenerate and fix translations
- 📊 **Detailed Logging**: View complete logs of SSH operations and verification results
- 🎨 **Modern UI**: Clean, responsive interface with syntax-highlighted code panels

## Quick Start

### Prerequisites

1. **Node.js** 18+ installed
2. **Ollama** running locally with `gemma3n:latest` model
3. **SSH access** to remote verification server (root@134.199.201.182)

### Installation & Running

```bash
# Install dependencies
npm install

# Run both frontend and backend
npm run dev:all
```

The app will be available at:
- Frontend: http://localhost:5173
- Backend API: http://localhost:3001

### Test Connection

```bash
# Test SSH connection to remote server
npm run test:ssh

# Check setup requirements
npm run test:setup
```

✅ **Verified**: SSH connection to the remote server (root@134.199.201.182) has been successfully tested.

## Detailed Setup

For comprehensive setup instructions, troubleshooting, and configuration details, see [SETUP.md](SETUP.md).

## How It Works

1. **Translation**: Enter source code → Click "Translate" → Ollama processes the code → View translated result
2. **Verification**: Click "Verify Kernel" → Backend SSHs to remote server → Activates Triton environment → Executes code → Returns results

## Technology Stack

- **Frontend**: React 19, TypeScript, Vite, TailwindCSS
- **Backend**: Express, Node.js, TypeScript
- **AI**: Ollama (gemma3n:latest model)
- **Remote Execution**: SSH2 for remote code verification

## Project Structure

```
local2/
├── server/              # Backend Express server
├── services/            # API client services
├── components/          # React components
├── App.tsx             # Main application
└── SETUP.md            # Detailed setup guide
```

## Documentation

- [SETUP.md](SETUP.md) - Complete setup and troubleshooting guide
- [API Documentation](SETUP.md#api-endpoints) - Backend API reference

## Original Version

View the original Gemini-based app: https://ai.studio/apps/drive/1N6GyUWoxcnTLP4k7gMMiZAL_hrXoWufo
