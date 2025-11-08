# Quick Start Guide

## Prerequisites Check

Run the setup checker:
```bash
./check-setup.sh
# or
npm run test:setup
```

Test SSH connection:
```bash
npm run test:ssh
```

✅ **SSH Test Passed**: The remote server connection has been successfully tested. See `SSH_TEST_RESULTS.md` for details.

## Installation

```bash
# 1. Install dependencies
npm install

# 2. Ensure Ollama is running with gemma3n:latest
ollama pull gemma3n:latest
ollama serve  # In a separate terminal if not already running

# 3. Verify SSH access (optional)
ssh -i ~/.ssh/id_ed25519 root@134.199.201.182
```

## Running the App

### Easy Way (Recommended)
```bash
npm run dev:all
```
This starts both frontend (port 5173) and backend (port 3001).

### Separate Terminals
```bash
# Terminal 1 - Backend
npm run server

# Terminal 2 - Frontend  
npm run dev
```

## Access the App

Open your browser to: **http://localhost:5173**

## How to Use

1. **Translate Code**
   - Enter CUDA/Triton/Mojo code in the left panel
   - Select source and target languages
   - Click "Translate" button
   - View translated code in the right panel

2. **Verify Code**
   - After translation, click "Verify Kernel" button
   - View compilation and execution results
   - See detailed SSH operation logs

3. **Regenerate with Feedback**
   - If translation has errors, click "Provide Feedback"
   - Paste compiler errors or describe issues
   - Click "Submit Feedback" to regenerate

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| Translation fails | Check if backend is running (`npm run server`) and Ollama is running (`ollama serve`) |
| "Model not found" | Run `ollama pull gemma3n:latest` |
| Verification fails | Check SSH key at `~/.ssh/id_ed25519` and remote server access |
| Port already in use | Kill the process on port 3001 or 5173 |

## Common Commands

```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# List available Ollama models
ollama list

# Test SSH connection
ssh -i ~/.ssh/id_ed25519 root@134.199.201.182 "echo test"

# Check what's running on ports
lsof -i :3001  # Backend
lsof -i :5173  # Frontend
```

## Need More Help?

- See [SETUP.md](SETUP.md) for detailed setup instructions
- See [CHANGES.md](CHANGES.md) for architecture details
- Check the console output for error messages

## Default Ports

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:3001
- **Ollama**: http://localhost:11434

## Example Workflow

```bash
# 1. Setup (one-time)
npm install
ollama pull gemma3n:latest

# 2. Start services
ollama serve &           # Start Ollama in background
npm run dev:all          # Start app

# 3. Use the app
# Open http://localhost:5173 in browser
# Translate some code
# Click verify to test on remote server

# 4. Stop services
# Ctrl+C in the terminal running npm
# killall ollama (if started in background)
```

That's it! You're ready to translate GPU kernels. 🚀

