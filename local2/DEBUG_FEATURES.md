# Debug Features - Packaged Script Saving & SSH Command Logging

## Overview

Two new debugging features have been added to help you troubleshoot verification issues:

1. **Packaged Scripts Saved Locally** - Every script Ollama creates is saved to your local machine
2. **Full SSH Commands Logged** - Complete SSH commands are shown in logs and activity panel

---

## 1. Local Script Debugging

### What Happens

When you click **"Verify Kernel"**, after Ollama packages your code into an executable Python script, that script is automatically saved to your local machine for inspection.

### File Location

```
~/kernel_packaged_debug/packaged_<timestamp>.py
```

**Example:**
```
/Users/yourusername/kernel_packaged_debug/packaged_1699472834567.py
```

### Directory Structure

```
~/ (your home directory)
└── kernel_packaged_debug/
    ├── packaged_1699472834567.py
    ├── packaged_1699472901234.py
    ├── packaged_1699473045789.py
    └── ...
```

### What You'll See in Logs

```
=== Step 1: Packaging Code with Ollama ===
Using model: gemma3n:latest
Language: Triton
Calling Ollama to package code...
✅ Code successfully packaged into executable script
Script size: 1847 characters
📝 Debug: Script saved to /Users/yourusername/kernel_packaged_debug/packaged_1699472834567.py
```

### Activity Log Entry

```
12:35:07 ► Debug File Saved
         📝 Packaged script saved to: /Users/yourusername/kernel_packaged_debug/packaged_1699472834567.py
```

### Why This Is Useful

1. **Inspect What Ollama Created** - See exactly what script is being sent to the remote server
2. **Debug Packaging Issues** - Check if Ollama added the right imports and structure
3. **Test Locally** - Copy the script and test it on your local machine
4. **Compare Versions** - Compare multiple packaged versions to see how Ollama changes things
5. **Manual Editing** - Edit the script and upload it manually if needed

### Example Packaged Script

```python
import torch
import triton
import triton.language as tl

print("EXECUTION START")
print("Setting up Triton kernel...")

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def run_kernel():
    size = 1024
    x = torch.randn(size, device='cuda')
    y = torch.randn(size, device='cuda')
    output = torch.zeros_like(x)
    
    print(f"Running kernel with {size} elements...")
    
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, size, BLOCK_SIZE=256)
    
    expected = x + y
    assert torch.allclose(output, expected), "Kernel output incorrect!"
    
    print("✓ Kernel executed successfully!")
    print(f"Sample results: {output[:5].tolist()}")

if __name__ == "__main__":
    try:
        run_kernel()
        print("EXECUTION COMPLETE")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
```

### Managing Debug Files

**View all debug files:**
```bash
ls -lh ~/kernel_packaged_debug/
```

**Open most recent file:**
```bash
# macOS
open ~/kernel_packaged_debug/packaged_*.py | tail -1

# Linux
xdg-open $(ls -t ~/kernel_packaged_debug/packaged_*.py | head -1)
```

**Clean up old debug files:**
```bash
# Remove files older than 7 days
find ~/kernel_packaged_debug/ -name "packaged_*.py" -mtime +7 -delete

# Remove all debug files
rm -rf ~/kernel_packaged_debug/
```

**Test a packaged script locally:**
```bash
cd ~/kernel_packaged_debug/
python packaged_1699472834567.py
```

---

## 2. SSH Command Logging

### What Happens

All SSH operations now log the **full command-line equivalent** of what's being executed, even though the actual connection uses the SSH2 library API.

### Where You'll See It

#### In Verification Logs

```
=== Step 2: Connecting to Remote Server ===
Target: root@134.199.201.182
SSH Key: ~/.ssh/id_ed25519
Full SSH command: ssh -i ~/.ssh/id_ed25519 root@134.199.201.182

=== Step 4: Executing Script on Remote Server ===
Activating Triton environment...
Command: source Triton-Puzzles/triton_env/bin/activate
Executing: python /tmp/kernel_verify_1699472834.py
Full command: ssh -i ~/.ssh/id_ed25519 root@134.199.201.182 "source Triton-Puzzles/triton_env/bin/activate && python /tmp/kernel_verify_1699472834.py"
```

#### In SSH Test Logs

```
=== Testing SSH Connection ===
Target: root@134.199.201.182
SSH Key: ~/.ssh/id_ed25519
Full SSH command: ssh -i ~/.ssh/id_ed25519 root@134.199.201.182

=== Executing test command: ls -la ===
Full command: ssh -i ~/.ssh/id_ed25519 root@134.199.201.182 "ls -la"
```

#### In Activity Log (UI)

```
12:35:07 ► Step 2: SSH Connection
         Using: ssh -i ~/.ssh/id_ed25519 root@134.199.201.182
```

### Why This Is Useful

1. **Reproduce Manually** - Copy/paste the exact command to run it yourself in terminal
2. **Debug SSH Issues** - Verify the correct key is being used
3. **Test Separately** - Run the command manually to see if it's an SSH issue or code issue
4. **Documentation** - Know exactly what commands are being run
5. **Security Audit** - Verify the connection details

### Manual Testing

You can copy any logged SSH command and run it manually:

**Test Connection:**
```bash
ssh -i ~/.ssh/id_ed25519 root@134.199.201.182
```

**Test ls Command:**
```bash
ssh -i ~/.ssh/id_ed25519 root@134.199.201.182 "ls -la"
```

**Test Triton Environment:**
```bash
ssh -i ~/.ssh/id_ed25519 root@134.199.201.182 "source Triton-Puzzles/triton_env/bin/activate && python --version"
```

**Run a Packaged Script Manually:**
```bash
# 1. Copy your local debug file to remote
scp -i ~/.ssh/id_ed25519 ~/kernel_packaged_debug/packaged_1699472834.py root@134.199.201.182:/tmp/

# 2. Execute it manually
ssh -i ~/.ssh/id_ed25519 root@134.199.201.182 "source Triton-Puzzles/triton_env/bin/activate && python /tmp/packaged_1699472834.py"
```

---

## How It Works Internally

### Script Saving (Backend)

```typescript
// After Ollama generates the script
const localScriptPath = path.join(os.homedir(), 'kernel_packaged_debug', `packaged_${timestamp}.py`);

// Create directory if needed
const debugDir = path.join(os.homedir(), 'kernel_packaged_debug');
if (!fs.existsSync(debugDir)) {
  fs.mkdirSync(debugDir, { recursive: true });
}

// Write script to file
fs.writeFileSync(localScriptPath, executableScript, 'utf8');
logs.push(`📝 Debug: Script saved to ${localScriptPath}`);
```

### SSH Command Logging (Backend)

```typescript
// Connection logging
logs.push(`Full SSH command: ssh -i ~/.ssh/id_ed25519 root@134.199.201.182`);

// Execution logging
logs.push(`Full command: ssh -i ~/.ssh/id_ed25519 root@134.199.201.182 "source Triton-Puzzles/triton_env/bin/activate && python ${remoteFileName}"`);
```

### Activity Log Integration (Frontend)

```typescript
// Extract debug file path from verification results
const debugFileMatch = result.match(/Debug: Script saved to (.+\.py)/);
if (debugFileMatch) {
  addActivity('info', 'Debug File Saved', `📝 Packaged script saved to: ${debugFileMatch[1]}`);
}

// Show SSH command
addActivity('info', 'Step 2: SSH Connection', 'Using: ssh -i ~/.ssh/id_ed25519 root@134.199.201.182');
```

---

## Common Debugging Scenarios

### Scenario 1: Verification Fails with Import Error

**Problem:** `ModuleNotFoundError: No module named 'triton'`

**Debug Steps:**
1. Check the packaged script in `~/kernel_packaged_debug/`
2. Look at the imports Ollama added
3. Test SSH connection manually: `ssh -i ~/.ssh/id_ed25519 root@134.199.201.182`
4. Test if Triton is available: `ssh -i ~/.ssh/id_ed25519 root@134.199.201.182 "source Triton-Puzzles/triton_env/bin/activate && python -c 'import triton; print(triton.__version__)'"`

### Scenario 2: Script Has Syntax Errors

**Problem:** `SyntaxError: invalid syntax`

**Debug Steps:**
1. Open the packaged script from `~/kernel_packaged_debug/`
2. Check for obvious syntax errors
3. Test locally: `python ~/kernel_packaged_debug/packaged_*.py` (if you have Triton locally)
4. Use "Provide Feedback" to regenerate with error message

### Scenario 3: Script Runs But Wrong Results

**Problem:** Code executes but produces incorrect output

**Debug Steps:**
1. Open the packaged script
2. Check the test data Ollama generated
3. Verify the kernel logic is correct
4. Modify the script locally and test
5. Compare with your original translated code

### Scenario 4: SSH Connection Issues

**Problem:** Can't connect to remote server

**Debug Steps:**
1. Check Activity Log for exact SSH command used
2. Copy the command and run manually in terminal
3. Check SSH key exists: `ls -la ~/.ssh/id_ed25519`
4. Check permissions: `chmod 600 ~/.ssh/id_ed25519`
5. Test basic connectivity: `ping 134.199.201.182`

---

## Benefits Summary

### For Debugging
- ✅ See exactly what Ollama generated
- ✅ Test scripts locally before remote execution
- ✅ Reproduce SSH commands manually
- ✅ Compare multiple packaging attempts

### For Learning
- ✅ Learn how Ollama structures Python scripts
- ✅ See proper Triton imports and patterns
- ✅ Understand remote execution flow
- ✅ Learn SSH best practices

### For Development
- ✅ Iterate faster on fixes
- ✅ Manually edit and test scripts
- ✅ Debug without re-running full pipeline
- ✅ Build test suites from working scripts

---

## Quick Reference

### File Locations
```
~/kernel_packaged_debug/          # All packaged scripts saved here
~/kernel_packaged_debug/packaged_<timestamp>.py  # Each verification creates one
```

### Commands
```bash
# View debug files
ls -lh ~/kernel_packaged_debug/

# Open latest file
open $(ls -t ~/kernel_packaged_debug/packaged_*.py | head -1)

# Test a script locally (if you have Triton)
python ~/kernel_packaged_debug/packaged_1699472834.py

# Copy to remote and test
scp -i ~/.ssh/id_ed25519 ~/kernel_packaged_debug/packaged_*.py root@134.199.201.182:/tmp/
ssh -i ~/.ssh/id_ed25519 root@134.199.201.182 "source Triton-Puzzles/triton_env/bin/activate && python /tmp/packaged_*.py"

# Clean up
rm -rf ~/kernel_packaged_debug/
```

### What to Look For in Packaged Scripts
- ✅ Correct imports (`import triton`, `import torch`, etc.)
- ✅ Kernel decorator (`@triton.jit`)
- ✅ Main execution block with test data
- ✅ Error handling (try/except)
- ✅ Print statements for debugging
- ✅ "EXECUTION START" and "EXECUTION COMPLETE" markers

---

## Summary

🎉 **New Debug Features:**

1. **📝 Script Saving** - Every packaged script saved to `~/kernel_packaged_debug/`
2. **🔍 SSH Logging** - Full SSH commands shown in logs and activity panel
3. **🔧 Manual Testing** - Copy commands and run them yourself
4. **📊 Activity Tracking** - See debug file path in Activity Log

Happy debugging! 🐛🔨

