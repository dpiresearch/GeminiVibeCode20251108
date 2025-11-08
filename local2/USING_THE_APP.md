# How to Use the GPU Kernel Translator

## Quick Start

```bash
# 1. Start the app
npm run dev:all

# 2. Open browser
open http://localhost:5173
```

## Complete Workflow

### 1. Test SSH Connection (Optional but Recommended)

**Why**: Verify remote server connectivity before verification

**How**:
1. Click **"Test SSH"** button (top-right corner)
2. Wait 2-3 seconds
3. See results:
   ```
   ✅ SSH connection established!
   ✅ Command executed successfully
   ✅ Triton-Puzzles directory found
   ```
4. Close panel when done

### 2. Translate Code

**Input Code**:
```cuda
// Example CUDA kernel
__global__ void add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

**Steps**:
1. Paste code in **left panel** (Source)
2. Select source language: **CUDA**
3. Select target language: **Triton**
4. Click **"Translate"** button
5. Wait for Ollama to process (5-10 seconds)
6. View translated code in **right panel**

**Output** (Triton):
```python
@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)
```

### 3. Verify Kernel (Real Execution!)

**What It Does**: 
- Packages code into executable Python script using Ollama
- SSHs to remote server
- Actually runs the code
- Returns real results

**Steps**:
1. After translation, click **"Verify Kernel"** button
2. Watch the 6-step process:

**Step 1: Packaging** (2-5 seconds)
```
=== Step 1: Packaging Code with Ollama ===
Using model: gemma3n:latest
Language: Triton
Calling Ollama to package code...
✅ Code successfully packaged into executable script
Script size: 1847 characters
```

**Step 2: Connecting** (0.5-1 second)
```
=== Step 2: Connecting to Remote Server ===
Target: root@134.199.201.182
SSH Key: ~/.ssh/id_ed25519
✅ SSH connection established
```

**Step 3: Uploading** (0.1-0.5 seconds)
```
=== Step 3: Uploading Script to Remote Server ===
Remote file: /tmp/kernel_verify_1699472834.py
✅ Script uploaded successfully
```

**Step 4: Executing** (2-10 seconds)
```
=== Step 4: Executing Script on Remote Server ===
Activating Triton environment...
Command: source Triton-Puzzles/triton_env/bin/activate
Executing: python /tmp/kernel_verify_1699472834.py
```

**Step 5: Results**
```
=== Step 5: Execution Results ===
Exit code: 0
✅ Execution completed successfully!

--- Standard Output (stdout) ---
EXECUTION START
Setting up Triton kernel...
Running kernel with 1024 elements...
✓ Kernel executed successfully!
Sample results: [1.234, 2.456, 3.789, 4.012, 5.345]
EXECUTION COMPLETE
```

**Step 6: Cleanup**
```
=== Step 6: Cleanup ===
Removing temporary file...
✅ Cleanup completed

=== Verification Complete ===
Final Status: SUCCESS
```

### 4. Handle Errors with Feedback

If verification fails or has issues:

1. Click **"Provide Feedback"** button
2. Paste error messages:
   ```
   Traceback (most recent call last):
     File "/tmp/kernel_verify_1699472834.py", line 15, in <module>
       import triton
   ModuleNotFoundError: No module named 'triton'
   ```
3. Or describe the issue:
   ```
   The kernel runs but produces incorrect results.
   Expected [1, 2, 3] but got [0, 0, 0].
   ```
4. Click **"Submit Feedback"**
5. Ollama regenerates the translation
6. New code appears in right panel
7. Verify again

## Real-World Example

### Scenario: Translate CUDA Vector Add to Triton

**1. Start with CUDA Code**
```cuda
__global__ void vector_add(float *a, float *b, float *c, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}
```

**2. Click Translate**
- Ollama processes locally
- ~5 seconds
- Returns Triton code

**3. Click Verify Kernel**
- Ollama packages into full script with:
  - Imports (`import torch, triton`)
  - Test data generation
  - Kernel launch code
  - Result verification
- SSH to server
- Actually executes
- Returns real output:
  ```
  EXECUTION START
  Running kernel with 1024 elements...
  ✓ Kernel executed successfully!
  EXECUTION COMPLETE
  ```

**4. Success!**
- Code works on real hardware
- No errors
- Ready for production use

## What Makes This Different?

### Before (Other Tools)
- ❌ Simulated verification
- ❌ AI guesses if code works
- ❌ No real execution
- ❌ Fake error messages

### Now (This App)
- ✅ Real execution on GPU server
- ✅ Actual compiler errors
- ✅ True stdout/stderr
- ✅ Validates code actually works
- ✅ Complete transparency (all logs)

## Tips & Tricks

### Tip 1: Test SSH First
Before translating lots of code, click "Test SSH" to ensure remote server is accessible.

### Tip 2: Read the Logs
The verification logs show everything that happens. If something fails, the logs tell you exactly where and why.

### Tip 3: Use Feedback Loop
Don't give up on first error! Use the feedback system:
1. Get error from verification
2. Paste into feedback
3. Regenerate
4. Verify again
5. Usually works on 2nd or 3rd try

### Tip 4: Start Simple
Test with simple kernels first (like vector add) before trying complex ones.

### Tip 5: Check Both Panels
- **Left Panel**: Your original code (editable)
- **Right Panel**: Translated code (read-only)
- **Verification Panel**: Execution results and logs

## Supported Languages

### Translation Pairs
- CUDA → Triton ✅
- CUDA → Mojo ✅
- Triton → CUDA ✅
- Triton → Mojo ✅
- Mojo → CUDA ✅
- Mojo → Triton ✅

### Verification
Currently optimized for **Triton** code execution on remote server.

## Performance Expectations

| Operation | Time | Notes |
|-----------|------|-------|
| Translation | 5-15 sec | Depends on code size |
| Verification | 5-17 sec | 6-step process |
| SSH Test | 2-3 sec | Quick connectivity check |
| Regeneration | 7-20 sec | Translation + feedback |

## Troubleshooting

### "Translation Failed"
**Cause**: Ollama not running or model not available

**Solution**:
```bash
# Start Ollama
ollama serve

# Pull model
ollama pull gemma3n:latest
```

### "Verification Failed: SSH Error"
**Cause**: Can't connect to remote server

**Solution**:
1. Click "Test SSH" to diagnose
2. Check SSH key exists: `ls ~/.ssh/id_ed25519`
3. Check permissions: `chmod 600 ~/.ssh/id_ed25519`
4. Test manually: `ssh -i ~/.ssh/id_ed25519 root@134.199.201.182`

### "Execution Failed: Module Not Found"
**Cause**: Package not available on remote server

**Solution**: Remote server needs the required packages installed (Triton, PyTorch, etc.)

### "Code Has Errors"
**Cause**: Translation produced incorrect code

**Solution**:
1. Click "Provide Feedback"
2. Paste error messages
3. Click "Submit Feedback"
4. Verify again

## Advanced Features

### Swap Languages
Click the swap icon (↔️) between panels to:
- Swap source and target languages
- Swap code between panels
- Useful for round-trip testing

### View Logs
All operations are logged:
- Frontend: Browser console (F12)
- Backend: Terminal running server
- Verification: Shown in UI

### Multiple Attempts
You can verify the same code multiple times:
- Each creates a new temporary file
- All cleaned up automatically
- All logs shown separately

## Best Practices

1. **Always test SSH** before heavy work
2. **Read verification logs** to understand what happened
3. **Use feedback** to improve translations
4. **Start simple** and build up complexity
5. **Check both panels** to compare code
6. **Save good translations** (copy/paste to your project)

## Summary

The GPU Kernel Translator gives you:
- ✅ Local AI translation (Ollama)
- ✅ Real remote verification (SSH + execution)
- ✅ Complete transparency (all logs)
- ✅ Feedback loop (regenerate with errors)
- ✅ Easy testing (Test SSH button)

Try it now! 🚀

```bash
npm run dev:all
# Then open http://localhost:5173
```

