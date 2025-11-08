# GPU Kernel Translator - Setup Guide

This application translates GPU kernels between CUDA, Triton, and Mojo using Ollama AI, and verifies the code on a remote server via SSH.

## Prerequisites

### 1. Ollama Installation
You need to have Ollama running locally with the `gemma3n:latest` model.

```bash
# Install Ollama (if not already installed)
# Visit: https://ollama.ai for installation instructions

# Pull the required model
ollama pull gemma3n:latest

# Start Ollama server (if not running)
ollama serve
```

Verify Ollama is running:
```bash
curl http://localhost:11434/api/version
```

### 2. SSH Configuration
The app connects to a remote server for code verification. Ensure you have:

- SSH key at `~/.ssh/id_ed25519`
- Access to the remote server: `root@134.199.201.182`
- The remote server should have the Triton environment at `Triton-Puzzles/triton_env/`

Test your SSH connection:
```bash
ssh -i ~/.ssh/id_ed25519 root@134.199.201.182 "echo 'Connection successful'"
```

### 3. Node.js
Ensure you have Node.js 18+ installed:
```bash
node --version
```

## Installation

1. Install dependencies:
```bash
npm install
```

## Running the Application

### Option 1: Run Both Frontend and Backend Together
```bash
npm run dev:all
```

This will start:
- Frontend (Vite) on `http://localhost:5173`
- Backend server on `http://localhost:3001`

### Option 2: Run Separately

**Terminal 1 - Backend Server:**
```bash
npm run server
```

**Terminal 2 - Frontend:**
```bash
npm run dev
```

## How It Works

### Translation Flow
1. User enters CUDA/Triton/Mojo code in the source panel
2. Click "Translate" button
3. Frontend sends request to backend server (`POST /api/translate`)
4. Backend calls Ollama locally with the `gemma3n:latest` model
5. Translated code appears in the translation panel

### Verification Flow
1. After translation, click "Verify Kernel" button
2. Frontend sends request to backend server (`POST /api/verify`)
3. Backend performs the following steps:
   - Establishes SSH connection to `root@134.199.201.182`
   - Creates a temporary file with the translated code
   - Activates the Triton environment: `source Triton-Puzzles/triton_env/bin/activate`
   - Executes the code and captures output
   - Returns compilation status, compiler output, and execution results
   - Cleans up temporary files
4. All logs and responses are displayed in the verification panel

## Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Check available models
ollama list

# If gemma3n:latest is not listed, pull it
ollama pull gemma3n:latest
```

### SSH Connection Issues
```bash
# Test SSH connection
ssh -i ~/.ssh/id_ed25519 root@134.199.201.182

# Check SSH key permissions (should be 600)
chmod 600 ~/.ssh/id_ed25519
```

### Backend Server Issues
```bash
# Check if port 3001 is available
lsof -i :3001

# Check backend logs
npm run server
```

### Frontend Issues
```bash
# Check if port 5173 is available
lsof -i :5173

# Clear cache and restart
rm -rf node_modules/.vite
npm run dev
```

## API Endpoints

### POST /api/translate
Translates code using Ollama.

**Request:**
```json
{
  "sourceLanguage": "CUDA",
  "targetLanguage": "Triton",
  "sourceCode": "...",
  "feedback": {
    "previousCode": "...",
    "userFeedback": "..."
  }
}
```

**Response:**
```json
{
  "translatedCode": "..."
}
```

### POST /api/verify
Verifies code on remote server via SSH.

**Request:**
```json
{
  "language": "Triton",
  "code": "..."
}
```

**Response:**
```json
{
  "result": "COMPILATION STATUS: SUCCESS\n\nCOMPILER OUTPUT:\n...\n\nEXECUTION OUTPUT:\n...\n\n=== DETAILED LOGS ===\n..."
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2025-11-08T..."
}
```

## Project Structure

```
local2/
├── server/
│   └── index.ts          # Backend server (Express + SSH)
├── services/
│   └── geminiService.ts  # API client for backend
├── components/
│   ├── CodePanel.tsx
│   ├── VerificationPanel.tsx
│   └── ...
├── App.tsx               # Main React component
├── package.json
├── tsconfig.json
├── tsconfig.server.json  # TypeScript config for server
└── SETUP.md             # This file
```

## Development Tips

- The backend uses `tsx watch` for hot-reloading during development
- Frontend uses Vite's fast HMR (Hot Module Replacement)
- Check browser console and terminal for error messages
- All SSH operations are logged in detail in the verification panel

## Security Notes

⚠️ **Important Security Considerations:**

1. The SSH private key is stored locally and never transmitted
2. The backend server should not be exposed to the public internet
3. Consider using environment variables for sensitive configuration
4. The remote server credentials are hardcoded - consider using environment variables in production

## Support

If you encounter issues:
1. Check that Ollama is running and has the `gemma3n:latest` model
2. Verify SSH connectivity to the remote server
3. Ensure all dependencies are installed (`npm install`)
4. Check both frontend and backend logs for errors

