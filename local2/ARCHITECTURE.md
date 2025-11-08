# Architecture Overview

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER BROWSER                            │
│                    http://localhost:5173                        │
│                                                                 │
│  ┌───────────────┐                    ┌──────────────────┐    │
│  │ Source Panel  │                    │ Translation Panel│    │
│  │  (editable)   │ ◄───────────────► │   (read-only)   │    │
│  │  CUDA/Triton/ │    Translate      │                  │    │
│  │    Mojo       │      Button       │  [Verify Kernel] │    │
│  └───────────────┘                    └──────────────────┘    │
│                                              │                 │
│                                              │ onClick         │
└──────────────────────────────────────────────┼─────────────────┘
                                               │
                    ┌──────────────────────────┼────────────────────┐
                    │                          │                    │
                    │   POST /api/translate    │  POST /api/verify  │
                    │                          │                    │
                    ▼                          ▼                    │
┌─────────────────────────────────────────────────────────────────┐
│              EXPRESS SERVER (Backend)                           │
│              http://localhost:3001                              │
│                                                                 │
│  ┌─────────────────────────┐  ┌───────────────────────────┐   │
│  │  Translation Handler    │  │   Verification Handler    │   │
│  │                         │  │                           │   │
│  │  1. Receive code        │  │  1. Receive code          │   │
│  │  2. Format prompt       │  │  2. SSH connect           │   │
│  │  3. Call Ollama         │  │  3. Upload code           │   │
│  │  4. Return result       │  │  4. Activate environment  │   │
│  │                         │  │  5. Execute code          │   │
│  └────────┬────────────────┘  │  6. Capture output        │   │
│           │                   │  7. Cleanup & return      │   │
│           │                   └──────────┬────────────────┘   │
└───────────┼────────────────────────────────┼──────────────────┘
            │                                │
            │ HTTP POST                      │ SSH Connection
            │                                │
            ▼                                ▼
┌────────────────────┐        ┌──────────────────────────────┐
│     OLLAMA         │        │      REMOTE SERVER           │
│  localhost:11434   │        │  root@134.199.201.182        │
│                    │        │                              │
│  gemma3n:latest    │        │  ┌────────────────────────┐ │
│     (AI Model)     │        │  │ Triton-Puzzles/        │ │
│                    │        │  │   triton_env/          │ │
│  Generates code    │        │  │                        │ │
│  translations      │        │  │ Activates via:         │ │
│                    │        │  │ source .../activate    │ │
└────────────────────┘        │  └────────────────────────┘ │
                              │                              │
                              │  Executes code and returns:  │
                              │  - stdout                    │
                              │  - stderr                    │
                              │  - exit code                 │
                              └──────────────────────────────┘
```

## Request Flow Diagrams

### Translation Flow

```
User Input
    │
    ├─ Enter CUDA/Triton/Mojo code
    ├─ Select source language
    ├─ Select target language
    │
    ▼
Click "Translate"
    │
    ▼
Frontend (React)
    │
    ├─ Validate input
    ├─ Show loading state
    │
    ▼
POST http://localhost:3001/api/translate
    │
    ├─ Body: {
    │     sourceLanguage: "CUDA",
    │     targetLanguage: "Triton",
    │     sourceCode: "...",
    │     feedback?: {...}
    │   }
    │
    ▼
Backend (Express)
    │
    ├─ Receive request
    ├─ Format prompt
    │
    ▼
POST http://localhost:11434/api/generate
    │
    ├─ Body: {
    │     model: "gemma3n:latest",
    │     prompt: "...",
    │     stream: false
    │   }
    │
    ▼
Ollama
    │
    ├─ Load gemma3n:latest model
    ├─ Process prompt
    ├─ Generate translation
    │
    ▼
Backend receives response
    │
    ├─ Extract translated code
    ├─ Return to frontend
    │
    ▼
Frontend displays result
    │
    └─ Show in translation panel
```

### Verification Flow

```
User Action
    │
    └─ Click "Verify Kernel"
         │
         ▼
Frontend (React)
    │
    ├─ Validate translated code exists
    ├─ Show verifying state
    │
    ▼
POST http://localhost:3001/api/verify
    │
    ├─ Body: {
    │     language: "Triton",
    │     code: "..."
    │   }
    │
    ▼
Backend (Express)
    │
    ├─ Initialize SSH client
    ├─ Log: "Starting SSH Connection"
    │
    ▼
SSH Connect to root@134.199.201.182
    │
    ├─ Use key: ~/.ssh/id_ed25519
    ├─ Timeout: 30s
    ├─ Log: "SSH connection established"
    │
    ▼
Create Remote File
    │
    ├─ Path: /tmp/kernel_<timestamp>.py
    ├─ Write code via: cat > file << 'EOF'
    ├─ Log: "File created successfully"
    │
    ▼
Execute Command
    │
    ├─ Command: source Triton-Puzzles/triton_env/bin/activate && 
    │           python /tmp/kernel_<timestamp>.py
    ├─ Log: "Executing verification command"
    │
    ▼
Capture Output
    │
    ├─ stdout → execution output
    ├─ stderr → compiler errors
    ├─ exit code → compilation status
    ├─ Log: "Execution completed"
    │
    ▼
Cleanup
    │
    ├─ rm -f /tmp/kernel_<timestamp>.py
    ├─ Log: "Cleanup completed"
    ├─ Close SSH connection
    │
    ▼
Format Response
    │
    ├─ COMPILATION STATUS: SUCCESS/FAILED
    ├─ COMPILER OUTPUT: ...
    ├─ EXECUTION OUTPUT: ...
    ├─ === DETAILED LOGS === ...
    │
    ▼
Return to Frontend
    │
    ▼
Display Results
    │
    ├─ Show compilation status
    ├─ Show compiler output
    ├─ Show execution output
    └─ Show all detailed logs
```

## Component Breakdown

### Frontend Components

```
App.tsx
  │
  ├─ CodePanel (Source)
  │   ├─ Language Selector
  │   ├─ Code Editor (editable)
  │   └─ Line Numbers
  │
  ├─ Swap Languages Button
  │
  ├─ CodePanel (Translation)
  │   ├─ Language Selector
  │   ├─ Code Editor (read-only)
  │   ├─ Verify Kernel Button
  │   └─ Provide Feedback Button
  │
  ├─ Feedback Panel (conditional)
  │   ├─ Textarea (for errors/feedback)
  │   ├─ Cancel Button
  │   └─ Submit Feedback Button
  │
  ├─ VerificationPanel (conditional)
  │   ├─ Loading Spinner
  │   ├─ Compilation Status
  │   ├─ Compiler Output
  │   ├─ Execution Output
  │   └─ Detailed Logs
  │
  └─ Translate Button (sticky footer)
```

### Backend Components

```
server/index.ts
  │
  ├─ Express App Setup
  │   ├─ CORS middleware
  │   ├─ JSON body parser
  │   └─ Logging middleware
  │
  ├─ Routes
  │   │
  │   ├─ POST /api/translate
  │   │   ├─ Request validation
  │   │   ├─ Prompt formatting
  │   │   ├─ Ollama API call
  │   │   └─ Response handling
  │   │
  │   ├─ POST /api/verify
  │   │   ├─ SSH connection setup
  │   │   ├─ File operations
  │   │   ├─ Command execution
  │   │   ├─ Output capture
  │   │   └─ Cleanup & response
  │   │
  │   └─ GET /health
  │       └─ Health check response
  │
  └─ Error Handling
      ├─ SSH errors
      ├─ Ollama errors
      └─ Generic errors
```

## Data Flow

### Translation Data Flow

```
User Code Input
    │
    ▼
{
  sourceLanguage: "CUDA",
  targetLanguage: "Triton",
  sourceCode: "__global__ void add(...) { ... }"
}
    │
    ▼
Backend formats prompt
    │
    ▼
"You are an expert AI... 
 Translate from CUDA to Triton...
 Source: __global__ void add(...) { ... }"
    │
    ▼
Ollama processes
    │
    ▼
{
  response: "import triton...\n@triton.jit\ndef add_kernel(...):\n..."
}
    │
    ▼
Backend extracts
    │
    ▼
{
  translatedCode: "import triton...\n@triton.jit\ndef add_kernel(...):\n..."
}
    │
    ▼
Frontend displays in translation panel
```

### Verification Data Flow

```
Translated Code
    │
    ▼
{
  language: "Triton",
  code: "import triton...\n@triton.jit\ndef add_kernel(...):\n..."
}
    │
    ▼
Backend SSH operations
    │
    ├─ Connect
    ├─ Write file
    ├─ Execute
    ├─ Capture
    └─ Cleanup
    │
    ▼
{
  logs: [
    "Starting SSH Connection",
    "SSH connection established",
    "Creating temporary file: /tmp/kernel_1699472234.py",
    "File created successfully",
    "Activating Triton environment",
    "Executing verification command",
    "STDOUT: ...",
    "STDERR: ...",
    "Execution completed (exit code: 0)",
    "Cleanup completed"
  ],
  compilationStatus: "SUCCESS",
  compilerOutput: "No errors.",
  executionOutput: "..."
}
    │
    ▼
Backend formats result
    │
    ▼
{
  result: "COMPILATION STATUS: SUCCESS\n\n
           COMPILER OUTPUT:\nNo errors.\n\n
           EXECUTION OUTPUT:\n...\n\n
           === DETAILED LOGS ===\n..."
}
    │
    ▼
Frontend displays in VerificationPanel
```

## Technology Stack

### Frontend
- **React 19**: UI framework
- **TypeScript**: Type safety
- **Vite**: Build tool & dev server
- **TailwindCSS**: Styling (from base app)

### Backend
- **Express**: Web server
- **Node.js**: Runtime
- **TypeScript**: Type safety
- **ssh2**: SSH client library

### AI & Execution
- **Ollama**: Local AI inference
- **gemma3n:latest**: Language model
- **SSH**: Remote code execution

## Port Configuration

| Service | Port | Access |
|---------|------|--------|
| Frontend (Vite) | 5173 | http://localhost:5173 |
| Backend (Express) | 3001 | http://localhost:3001 |
| Ollama API | 11434 | http://localhost:11434 |
| Remote SSH | 22 | 134.199.201.182:22 |

## File Structure

```
local2/
├── server/
│   └── index.ts              # Backend Express server
│
├── services/
│   └── geminiService.ts      # API client (calls backend)
│
├── components/
│   ├── CodePanel.tsx         # Source & translation panels
│   ├── VerificationPanel.tsx # Verification results
│   ├── LoadingSpinner.tsx    # Loading indicator
│   └── ArrowIcon.tsx         # UI icon
│
├── App.tsx                   # Main React component
├── index.tsx                 # React entry point
├── types.ts                  # TypeScript types
├── constants.ts              # App constants
│
├── package.json              # Dependencies & scripts
├── tsconfig.json             # TS config (frontend)
├── tsconfig.server.json      # TS config (backend)
├── vite.config.ts            # Vite configuration
│
├── README.md                 # Project overview
├── SETUP.md                  # Detailed setup guide
├── QUICKSTART.md             # Quick reference
├── CHANGES.md                # Change documentation
├── ARCHITECTURE.md           # This file
├── IMPLEMENTATION_SUMMARY.md # Summary of implementation
│
├── check-setup.sh            # Setup verification script
└── .gitignore                # Git ignore rules
```

## Security Considerations

### SSH Security
- Private key stored locally at `~/.ssh/id_ed25519`
- Key never transmitted over network
- SSH connection uses key-based auth (no passwords)
- Recommended key permissions: 600 (read/write for owner only)

### Server Security
- Backend runs locally (localhost only)
- Not exposed to public internet
- CORS enabled for frontend access
- No authentication currently implemented

### Code Execution
- Code executed in isolated remote environment
- Temporary files created with unique timestamps
- Files cleaned up after execution
- Remote environment managed separately

### Recommendations for Production
1. Add authentication to backend API
2. Use environment variables for sensitive config
3. Implement rate limiting
4. Add request validation and sanitization
5. Use HTTPS for production deployment
6. Implement SSH connection pooling
7. Add monitoring and logging

## Performance Characteristics

### Translation
- **Speed**: Depends on local hardware and model size
- **Memory**: Ollama needs sufficient RAM for model
- **CPU/GPU**: Faster with GPU acceleration

### Verification
- **Network Latency**: 50-200ms to remote server
- **SSH Overhead**: ~1-2 seconds for connection/setup
- **Execution Time**: Depends on code complexity
- **Total Time**: Typically 2-10 seconds

### Scalability
- **Concurrent Translations**: Limited by Ollama capacity
- **Concurrent Verifications**: Limited by SSH connections
- **Recommended**: Run tasks sequentially

## Error Handling

### Frontend Errors
- Empty source code
- Same source/target language
- Network errors
- Invalid responses

### Backend Errors
- Ollama not running
- Model not found
- SSH connection failures
- Remote execution errors
- Timeout errors

### Recovery Mechanisms
- Graceful error messages
- Automatic cleanup on failure
- Connection timeout handling
- Detailed error logging

## Monitoring & Debugging

### Logs Location
- **Frontend**: Browser console (F12)
- **Backend**: Terminal running `npm run server`
- **Ollama**: Ollama server logs
- **SSH**: Included in verification response

### Debug Commands
```bash
# Check backend health
curl http://localhost:3001/health

# Check Ollama
curl http://localhost:11434/api/version

# Test SSH
ssh -i ~/.ssh/id_ed25519 root@134.199.201.182 "echo test"

# View backend logs (if using systemd)
journalctl -u gpu-translator -f
```

## Future Enhancements

1. **Real-time Streaming**
   - Stream translation progress
   - Live SSH output

2. **Multi-Server Support**
   - Verify on multiple platforms
   - Parallel execution

3. **Caching**
   - Cache translations
   - Cache verification results

4. **Enhanced UI**
   - Side-by-side diff view
   - Syntax highlighting improvements
   - Export results

5. **Advanced Features**
   - Performance profiling
   - Benchmark comparisons
   - A/B testing different models

This architecture provides a robust, secure, and scalable foundation for GPU kernel translation and verification.

