# SCP/SSH Update - Command-Line Based Verification

## Overview

The verification system has been updated to use **command-line `scp` and `ssh`** tools instead of the SSH2 library's API for more control and better debugging.

---

## What Changed

### Before (SSH2 Library)
- Used SSH2 `conn.exec()` with heredoc to write files
- All operations through SSH2 library API
- File uploaded via: `cat > file << 'EOF'...`

### After (Command-Line Tools)
- Uses `scp` command to copy files
- Uses `ssh` command to execute scripts
- Better matches manual workflow
- Easier to debug with exact commands

---

## New Flow

### Step 1: Package with Ollama ✅
- Ollama creates executable Python script
- **Strips markdown code fences** (```python and ```)
- Saves to: `~/kernel_packaged_debug/packaged_<timestamp>.py`

### Step 2: Check SSH Key ✅
- Verifies `~/.ssh/id_ed25519` exists
- Shows full path in logs

### Step 3: Upload via SCP ✅
- **Uses `scp` command to copy file**
- Command: `scp -i ~/.ssh/id_ed25519 /path/to/packaged.py root@134.199.201.182:~/`
- Uploads to remote home directory (`~`)

### Step 4: Execute via SSH ✅
- **Uses `ssh` command to run script**
- Command: `ssh -i ~/.ssh/id_ed25519 root@134.199.201.182 'source Triton-Puzzles/triton_env/bin/activate; python3 ~/packaged_<timestamp>.py'`
- Uses **python3** (not python)
- Sources Triton environment first

### Step 5: Results ✅
- Captures stdout, stderr, exit code
- Shows all output in logs

### Step 6: Cleanup ✅
- Removes remote file: `ssh -i ~/.ssh/id_ed25519 root@134.199.201.182 'rm -f ~/packaged_<timestamp>.py'`

---

## Key Improvements

### 1. ✅ Markdown Fence Stripping

```typescript
executableScript = executableScript
  .replace(/^```python\s*/i, '')  // Remove leading ```python
  .replace(/^```\s*/i, '')         // Remove leading ```
  .replace(/\s*```\s*$/i, '')      // Remove trailing ```
  .trim();
```

**Why**: Ollama sometimes wraps code in markdown fences. These would break the Python script.

### 2. ✅ SCP Upload

```bash
scp -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no /path/to/packaged.py root@134.199.201.182:~/
```

**Benefits**:
- Direct file copy (faster)
- No escaping issues with code content
- Exact same command you'd use manually
- `-o StrictHostKeyChecking=no` avoids host key prompts

### 3. ✅ SSH Execution

```bash
ssh -i ~/.ssh/id_ed25519 root@134.199.201.182 'source Triton-Puzzles/triton_env/bin/activate; python3 ~/packaged_<timestamp>.py'
```

**Benefits**:
- Uses `python3` explicitly (more modern)
- Single-quote string (no escaping needed)
- Semicolon separator (works in any shell)
- Exact command shown in logs

### 4. ✅ Better Error Handling

```typescript
try {
  const { stdout, stderr } = await execAsync(sshCommand);
  exitCode = 0;
} catch (execError: any) {
  stdout = execError.stdout || '';
  stderr = execError.stderr || '';
  exitCode = execError.code || 1;
}
```

**Benefits**:
- Captures stdout/stderr even on failure
- Shows exact exit code
- No data loss on errors

---

## Example Logs

### Successful Verification

```
=== Step 1: Packaging Code with Ollama ===
Using model: gemma3n:latest
Language: Triton
Calling Ollama to package code...
✅ Code successfully packaged into executable script
Script size: 1847 characters
Markdown code fences stripped (if present)
📝 Debug: Script saved to /Users/dpang/kernel_packaged_debug/packaged_1762642962017.py

=== Step 2: Preparing SSH Connection ===
SSH Key: /Users/dpang/.ssh/id_ed25519
✅ SSH key found

=== Step 3: Uploading Script via SCP ===
Local file: /Users/dpang/kernel_packaged_debug/packaged_1762642962017.py
Remote destination: root@134.199.201.182:~/packaged_1762642962017.py
SCP command: scp -i /Users/dpang/.ssh/id_ed25519 -o StrictHostKeyChecking=no /Users/dpang/kernel_packaged_debug/packaged_1762642962017.py root@134.199.201.182:~/packaged_1762642962017.py
✅ Script uploaded successfully via SCP

=== Step 4: Executing Script on Remote Server ===
Activating Triton environment and running script...
Remote script: ~/packaged_1762642962017.py
SSH command: ssh -i /Users/dpang/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@134.199.201.182 'source Triton-Puzzles/triton_env/bin/activate; python3 ~/packaged_1762642962017.py'

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

=== Step 6: Cleanup ===
Removing temporary file from remote server...
Cleanup command: ssh -i /Users/dpang/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@134.199.201.182 'rm -f ~/packaged_1762642962017.py'
✅ Cleanup completed

=== Verification Complete ===
Final Status: SUCCESS
```

---

## Manual Testing

You can now easily reproduce any verification manually:

### 1. Copy the exact commands from logs

From the logs above, copy these commands:

**Upload:**
```bash
scp -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no ~/kernel_packaged_debug/packaged_1762642962017.py root@134.199.201.182:~/packaged_1762642962017.py
```

**Execute:**
```bash
ssh -i ~/.ssh/id_ed25519 root@134.199.201.182 'source Triton-Puzzles/triton_env/bin/activate; python3 ~/packaged_1762642962017.py'
```

**Cleanup:**
```bash
ssh -i ~/.ssh/id_ed25519 root@134.199.201.182 'rm -f ~/packaged_1762642962017.py'
```

### 2. Test step by step

```bash
# 1. Check local file exists
ls -lh ~/kernel_packaged_debug/packaged_*.py

# 2. Upload with scp
scp -i ~/.ssh/id_ed25519 ~/kernel_packaged_debug/packaged_1762642962017.py root@134.199.201.182:~/

# 3. Verify it's there
ssh -i ~/.ssh/id_ed25519 root@134.199.201.182 'ls -lh ~/packaged_*.py'

# 4. Run it
ssh -i ~/.ssh/id_ed25519 root@134.199.201.182 'source Triton-Puzzles/triton_env/bin/activate; python3 ~/packaged_1762642962017.py'

# 5. Check output
# (see stdout from previous command)

# 6. Clean up
ssh -i ~/.ssh/id_ed25519 root@134.199.201.182 'rm ~/packaged_*.py'
```

---

## Technical Details

### File Paths

| Location | Path | Example |
|----------|------|---------|
| Local debug | `~/kernel_packaged_debug/packaged_<timestamp>.py` | `/Users/dpang/kernel_packaged_debug/packaged_1762642962017.py` |
| Remote home | `~/packaged_<timestamp>.py` | `/root/packaged_1762642962017.py` |

### Commands Used

| Tool | Purpose | Flags Used |
|------|---------|------------|
| `scp` | Upload file | `-i` (key), `-o StrictHostKeyChecking=no` |
| `ssh` | Execute script | `-i` (key), `-o StrictHostKeyChecking=no` |
| `python3` | Run Python | (standard) |

### Why python3?

Modern systems have both `python` and `python3`:
- `python` might be Python 2 (deprecated)
- `python3` is guaranteed to be Python 3.x
- Triton requires Python 3.6+

### Why Semicolon in SSH Command?

```bash
'source ...; python3 ...'  # ✅ Works in any shell
'source ... && python3 ...' # ❌ Bash-specific
```

Semicolon (`;`) works in all shells, `&&` is bash-specific.

---

## Benefits Summary

### For Users
✅ **Exact commands in logs** - Copy/paste to reproduce  
✅ **Easier debugging** - Test each step manually  
✅ **Better error messages** - Full stdout/stderr captured  
✅ **No code fences** - Markdown stripped automatically  

### For Developers
✅ **Simpler code** - Uses standard CLI tools  
✅ **Better control** - Direct command execution  
✅ **Easier testing** - Run commands directly  
✅ **Standard workflow** - Matches manual SSH usage  

---

## Backward Compatibility

### Test SSH Endpoint
- Still uses SSH2 library (unchanged)
- Works exactly as before
- Useful for quick connection tests

### Verification Endpoint
- Now uses CLI tools (`scp`, `ssh`)
- More robust and debuggable
- Shows exact commands in logs

---

## Troubleshooting

### SCP Upload Fails

**Error**: `Permission denied (publickey)`

**Fix**:
```bash
# Check key permissions
chmod 600 ~/.ssh/id_ed25519

# Test SSH access first
ssh -i ~/.ssh/id_ed25519 root@134.199.201.182 'echo test'
```

### SSH Execution Fails

**Error**: `Command not found: python3`

**Fix**:
```bash
# Check what's available
ssh -i ~/.ssh/id_ed25519 root@134.199.201.182 'which python python3'

# If needed, use python instead of python3 (edit server code)
```

### Markdown Fences Not Stripped

**Symptom**: Script fails with `SyntaxError: invalid syntax`

**Check**: Look at the debug file:
```bash
head -5 ~/kernel_packaged_debug/packaged_*.py
# Should show: import torch
# Should NOT show: ```python
```

If still present, the stripping logic needs adjustment.

---

## Summary

🎉 **Verification now uses standard CLI tools!**

- ✅ `scp` for file upload
- ✅ `ssh` for execution  
- ✅ Markdown fences stripped
- ✅ `python3` used
- ✅ Exact commands logged
- ✅ Easy manual reproduction

The workflow is now identical to how you'd do it manually, making debugging much easier! 🚀

