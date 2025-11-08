import express, { Request, Response } from 'express';
import cors from 'cors';
import { Client } from 'ssh2';
import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs';
import path from 'path';
import os from 'os';

// Note: SSH2 Client is still used for the Test SSH endpoint
// Verification endpoint now uses command-line scp and ssh for better control

const execAsync = promisify(exec);

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Logging middleware
app.use((req, _res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
  next();
});

interface TranslateRequest {
  sourceLanguage: string;
  targetLanguage: string;
  sourceCode: string;
  feedback?: {
    previousCode: string;
    userFeedback: string;
  };
}

interface VerifyRequest {
  language: string;
  code: string;
  feedback?: {
    previousScript: string;
    userFeedback: string;
    previousError: string;
  };
}

// Translation endpoint using Ollama
app.post('/api/translate', async (req: Request, res: Response) => {
  try {
    const { sourceLanguage, targetLanguage, sourceCode, feedback }: TranslateRequest = req.body;

    console.log(`Translation request: ${sourceLanguage} -> ${targetLanguage}`);
    if (feedback) {
      console.log('🔄 Regeneration with feedback requested');
      console.log(`Previous code length: ${feedback.previousCode.length}`);
      console.log(`Feedback length: ${feedback.userFeedback.length}`);
    }

    let prompt: string;

    if (feedback) {
      prompt = `You are an expert AI assistant specializing in high-performance GPU computing.
You previously attempted to translate a GPU kernel from ${sourceLanguage} to ${targetLanguage}, but the generated code had issues.

CRITICAL INSTRUCTIONS:
1. Carefully analyze the user's feedback to understand what went wrong
2. Do NOT repeat the same mistakes from the previous attempt
3. Fix ALL issues mentioned in the user feedback
4. Generate working, correct code this time
5. If the feedback mentions specific errors, address each one explicitly

Original Source Code (${sourceLanguage}):
---
${sourceCode}
---

Previous Translation That Had Issues (${targetLanguage}):
---
${feedback.previousCode}
---

User Feedback / Compiler Errors / Issues to Fix:
---
${feedback.userFeedback}
---

IMPORTANT: The user has identified specific problems. Make sure your corrected translation:
- Fixes ALL the issues mentioned in the feedback above
- Does NOT repeat any of the mistakes from the previous attempt
- Includes all necessary imports and dependencies
- Uses correct syntax and API calls for ${targetLanguage}
- Is executable and will run without errors

Corrected Code (${targetLanguage}):
(Output ONLY the complete, corrected code. No explanations, no comments about changes, no markdown code fences like \`\`\`. Just pure ${targetLanguage} code that works.)`;
    } else {
      prompt = `You are an expert AI assistant specializing in high-performance GPU computing.
Your task is to translate a GPU kernel from ${sourceLanguage} to ${targetLanguage}.
Focus on maintaining the original algorithm's logic, performance optimizations, and memory access patterns.
The output must be only the raw, complete, and valid code in ${targetLanguage}.
Do not include any explanations, comments about the translation, or markdown code fences like \`\`\`${targetLanguage.toLowerCase()}\`\`\`.

Source Code (${sourceLanguage}):
---
${sourceCode}
---

Translated Code (${targetLanguage}):`;
    }

    // Call Ollama API
    const ollamaResponse = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gemma3n:latest',
        prompt: prompt,
        stream: false,
      }),
    });

    if (!ollamaResponse.ok) {
      throw new Error(`Ollama API error: ${ollamaResponse.status} ${ollamaResponse.statusText}`);
    }

    const ollamaData = await ollamaResponse.json();
    const translatedCode = ollamaData.response.trim();

    console.log('Translation completed successfully');
    res.json({ translatedCode });
  } catch (error) {
    console.error('Translation error:', error);
    res.status(500).json({
      error: error instanceof Error ? error.message : 'Failed to translate code',
    });
  }
});

// Verification endpoint using SSH
app.post('/api/verify', async (req: Request, res: Response) => {
  const { language, code, feedback }: VerifyRequest = req.body;
  
  console.log(`Verification request for ${language} code`);
  console.log('Code to verify:', code.substring(0, 100) + '...');
  
  if (feedback) {
    console.log('🔄 Re-verification with feedback requested');
    console.log(`Previous script length: ${feedback.previousScript.length}`);
    console.log(`Feedback length: ${feedback.userFeedback.length}`);
    console.log(`Previous error length: ${feedback.previousError.length}`);
  }

  const logs: string[] = [];
  let compilationStatus = 'UNKNOWN';
  let compilerOutput = '';
  let executionOutput = '';
  
  // Variables for file paths (declared at function scope)
  let localScriptPath = '';
  let remoteScriptName = '';

  try {
    // Step 1: Use Ollama to package code into executable Python script
    logs.push('=== Step 1: Packaging Code with Ollama ===');
    logs.push(`Using model: gemma3n:latest`);
    logs.push(`Language: ${language}`);
    if (feedback) {
      logs.push('🔄 Incorporating feedback from previous attempt');
    }
    logs.push('');

    let packagingPrompt: string;
    
    if (feedback) {
      packagingPrompt = `You are an expert in GPU programming and Python.

You previously attempted to package a ${language} kernel into an executable Python script, but it had issues when executed.

CRITICAL INSTRUCTIONS FOR RE-PACKAGING:
1. Carefully analyze the previous execution error to understand what went wrong
2. Do NOT repeat the same mistakes from the previous packaged script
3. Fix ALL issues mentioned in the user feedback and execution errors
4. Generate a working, executable script this time
5. Address each error explicitly

${language} Kernel Code to Package:
---
${code}
---

Previous Packaged Script That Had Issues:
---
${feedback.previousScript}
---

Execution Error / Issues from Previous Attempt:
---
${feedback.previousError}
---

User Feedback:
---
${feedback.userFeedback}
---

IMPORTANT: The previous script failed when executed. Make sure your new packaged script:
- Fixes ALL the issues mentioned above (missing imports, syntax errors, incorrect API usage, etc.)
- Does NOT repeat any of the mistakes from the previous attempt
- Includes ALL necessary imports and dependencies at the top
- Uses correct syntax and API calls for ${language}
- Has proper error handling
- Will actually run without errors when executed with: python script.py
- Prints "EXECUTION START" at the beginning and "EXECUTION COMPLETE" at the end

Requirements:
1. Add all necessary imports at the top (don't forget any!)
2. If this is Triton code, ensure proper Triton imports and decorators
3. Add a main execution block that demonstrates the kernel with sample data
4. Include error handling and print statements showing execution progress
5. Make it runnable with: python script.py
6. Output ONLY the complete Python script, no explanations or markdown

Complete Corrected Executable Python Script:
(Output ONLY the complete, working Python script. No explanations, no comments about changes, no markdown code fences.)`;
    } else {
      packagingPrompt = `You are an expert in GPU programming and Python.

Your task is to take the following ${language} kernel code and package it into a complete, executable Python script.

Requirements:
1. Add all necessary imports at the top
2. If this is Triton code, ensure proper Triton imports and decorators
3. Add a main execution block that demonstrates the kernel with sample data
4. Include error handling and print statements showing execution progress
5. Make it runnable with: python script.py
6. The script should print "EXECUTION START" at the beginning and "EXECUTION COMPLETE" at the end
7. Output ONLY the complete Python script, no explanations or markdown

${language} Kernel Code:
---
${code}
---

Complete Executable Python Script:`;
    }

    let executableScript = '';
    
    try {
      logs.push('Calling Ollama to package code...');
      const ollamaResponse = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'gemma3n:latest',
          prompt: packagingPrompt,
          stream: false,
        }),
      });

      if (!ollamaResponse.ok) {
        throw new Error(`Ollama API error: ${ollamaResponse.status}`);
      }

      const ollamaData = await ollamaResponse.json();
      executableScript = ollamaData.response.trim();
      
      // Strip markdown code fences if present
      executableScript = executableScript
        .replace(/^```python\s*/i, '')  // Remove leading ```python
        .replace(/^```\s*/i, '')         // Remove leading ```
        .replace(/\s*```\s*$/i, '')      // Remove trailing ```
        .trim();
      
      logs.push('✅ Code successfully packaged into executable script');
      logs.push(`Script size: ${executableScript.length} characters`);
      logs.push('Markdown code fences stripped (if present)');
      
      // Write packaged script to local file for debugging
      // Use readable date format: MMDD_HHMMSS
      const now = new Date();
      const month = String(now.getMonth() + 1).padStart(2, '0');
      const day = String(now.getDate()).padStart(2, '0');
      const hours = String(now.getHours()).padStart(2, '0');
      const minutes = String(now.getMinutes()).padStart(2, '0');
      const seconds = String(now.getSeconds()).padStart(2, '0');
      const dateTimeStr = `${month}${day}_${hours}${minutes}${seconds}`;
      
      localScriptPath = path.join(os.homedir(), 'kernel_packaged_debug', `packaged_${dateTimeStr}.py`);
      remoteScriptName = `packaged_${dateTimeStr}.py`;
      
      logs.push(`Filename: packaged_${dateTimeStr}.py`);
      
      try {
        // Create directory if it doesn't exist
        const debugDir = path.join(os.homedir(), 'kernel_packaged_debug');
        if (!fs.existsSync(debugDir)) {
          fs.mkdirSync(debugDir, { recursive: true });
        }
        
        // Write script to file
        fs.writeFileSync(localScriptPath, executableScript, 'utf8');
        logs.push(`📝 Debug: Script saved to ${localScriptPath}`);
        console.log(`Debug script saved: ${localScriptPath}`);
      } catch (writeError) {
        logs.push(`⚠️  Warning: Could not save debug file: ${writeError instanceof Error ? writeError.message : 'Unknown error'}`);
      }
      
      logs.push('');
    } catch (ollamaError) {
      logs.push(`❌ Ollama packaging failed: ${ollamaError instanceof Error ? ollamaError.message : 'Unknown error'}`);
      throw ollamaError;
    }

    // Step 2: Check SSH key exists
    logs.push('=== Step 2: Preparing SSH Connection ===');
    const sshKeyPath = path.join(os.homedir(), '.ssh', 'id_ed25519');
    logs.push(`SSH Key: ${sshKeyPath}`);
    
    if (!fs.existsSync(sshKeyPath)) {
      logs.push(`❌ SSH key not found at ${sshKeyPath}`);
      throw new Error(`SSH key not found at ${sshKeyPath}`);
    }
    logs.push('✅ SSH key found');
    logs.push('');

    // Step 3: Upload script using scp
    logs.push('=== Step 3: Uploading Script via SCP ===');
    logs.push(`Local file: ${localScriptPath}`);
    logs.push(`Remote destination: root@134.199.201.182:~/${remoteScriptName}`);
    
    const scpCommand = `scp -i ${sshKeyPath} -o StrictHostKeyChecking=no ${localScriptPath} root@134.199.201.182:~/${remoteScriptName}`;
    logs.push(`SCP command: ${scpCommand}`);
    logs.push('');

    try {
      const { stdout: scpStdout, stderr: scpStderr } = await execAsync(scpCommand);
      logs.push('✅ Script uploaded successfully via SCP');
      if (scpStdout) logs.push(`SCP output: ${scpStdout}`);
      if (scpStderr) logs.push(`SCP info: ${scpStderr}`);
      logs.push('');
    } catch (scpError) {
      logs.push(`❌ SCP upload failed: ${scpError instanceof Error ? scpError.message : 'Unknown error'}`);
      throw scpError;
    }

    // Step 4: Execute the script on remote server via SSH
    logs.push('=== Step 4: Executing Script on Remote Server ===');
    logs.push('Activating Triton environment and running script...');
    logs.push(`Remote script: ~/${remoteScriptName}`);
    
    const sshCommand = `ssh -i ${sshKeyPath} -o StrictHostKeyChecking=no root@134.199.201.182 'source Triton-Puzzles/triton_env/bin/activate; python3 ~/${remoteScriptName}'`;
    logs.push(`SSH command: ${sshCommand}`);
    logs.push('');

    let stdout = '';
    let stderr = '';
    let exitCode = 0;

    try {
      const { stdout: execStdout, stderr: execStderr } = await execAsync(sshCommand);
      stdout = execStdout;
      stderr = execStderr;
      exitCode = 0;
    } catch (execError: any) {
      // execAsync throws on non-zero exit code
      stdout = execError.stdout || '';
      stderr = execError.stderr || '';
      exitCode = execError.code || 1;
    }

    logs.push('=== Step 5: Execution Results ===');
    logs.push(`Exit code: ${exitCode}`);
    logs.push('');
    
    if (exitCode === 0) {
      compilationStatus = 'SUCCESS';
      compilerOutput = 'No errors.';
      executionOutput = stdout || 'No output';
      logs.push('✅ Execution completed successfully!');
    } else {
      compilationStatus = 'FAILED';
      compilerOutput = stderr || 'Unknown error occurred';
      executionOutput = 'N/A';
      logs.push('❌ Execution failed');
    }

    logs.push('');
    if (stdout) {
      logs.push('--- Standard Output (stdout) ---');
      logs.push(stdout);
      logs.push('');
    }
    if (stderr) {
      logs.push('--- Standard Error (stderr) ---');
      logs.push(stderr);
      logs.push('');
    }

    // Step 6: Cleanup
    logs.push('=== Step 6: Cleanup ===');
    logs.push('Removing temporary file from remote server...');
    
    const cleanupCommand = `ssh -i ${sshKeyPath} -o StrictHostKeyChecking=no root@134.199.201.182 'rm -f ~/${remoteScriptName}'`;
    logs.push(`Cleanup command: ${cleanupCommand}`);
    
    try {
      await execAsync(cleanupCommand);
      logs.push('✅ Cleanup completed');
    } catch (cleanupError) {
      logs.push(`⚠️  Warning: Failed to remove temporary file: ${cleanupError instanceof Error ? cleanupError.message : 'Unknown error'}`);
    }
    logs.push('');

    logs.push('=== Verification Complete ===');
    logs.push(`Final Status: ${compilationStatus}`);

    const result = `
COMPILATION STATUS: ${compilationStatus}

COMPILER OUTPUT:
${compilerOutput}

EXECUTION OUTPUT:
${executionOutput}

=== DETAILED LOGS ===
${logs.join('\n')}
    `.trim();

    console.log('✅ Verification completed successfully');
    res.json({ 
      result,
      packagedScript: executableScript,  // Include packaged script for feedback
      executionError: stderr || ''       // Include error for feedback
    });
  } catch (error) {
    logs.push('');
    logs.push('=== ERROR ===');
    logs.push(`❌ ${error instanceof Error ? error.message : 'Unknown error'}`);

    const result = `
COMPILATION STATUS: FAILED

COMPILER OUTPUT:
Error during verification: ${error instanceof Error ? error.message : 'Unknown error'}

EXECUTION OUTPUT:
N/A

=== DETAILED LOGS ===
${logs.join('\n')}
    `.trim();

    console.error('❌ Verification error:', error);
    res.json({ result });
  }
});

// Test SSH connection endpoint
app.post('/api/test-ssh', async (_req: Request, res: Response) => {
  console.log('SSH connection test requested');
  
  const conn = new Client();
  const logs: string[] = [];
  
  try {
    logs.push('=== Testing SSH Connection ===');
    logs.push(`Target: root@134.199.201.182`);
    logs.push(`SSH Key: ~/.ssh/id_ed25519`);
    logs.push(`Full SSH command: ssh -i ~/.ssh/id_ed25519 root@134.199.201.182`);
    logs.push('');
    
    await new Promise<void>((resolve, reject) => {
      const sshKeyPath = path.join(os.homedir(), '.ssh', 'id_ed25519');
      
      if (!fs.existsSync(sshKeyPath)) {
        reject(new Error(`SSH key not found at ${sshKeyPath}`));
        return;
      }
      
      const privateKey = fs.readFileSync(sshKeyPath);
      
      conn.on('ready', () => {
        logs.push('✅ SSH connection established successfully!');
        resolve();
      });
      
      conn.on('error', (err) => {
        logs.push(`❌ SSH connection error: ${err.message}`);
        reject(err);
      });
      
      conn.connect({
        host: '134.199.201.182',
        port: 22,
        username: 'root',
        privateKey: privateKey,
        readyTimeout: 30000,
      });
    });
    
    // Execute ls -la command to verify
    logs.push('');
    logs.push('=== Executing test command: ls -la ===');
    logs.push(`Full command: ssh -i ~/.ssh/id_ed25519 root@134.199.201.182 "ls -la"`);
    
    let commandOutput = '';
    
    await new Promise<void>((resolve, reject) => {
      conn.exec('ls -la', (err, stream) => {
        if (err) {
          reject(err);
          return;
        }
        
        let stdout = '';
        let stderr = '';
        
        stream.on('close', (exitCode: number) => {
          if (exitCode === 0) {
            logs.push('✅ Command executed successfully');
            logs.push('');
            logs.push('Output:');
            logs.push(stdout);
            commandOutput = stdout;
            resolve();
          } else {
            logs.push(`❌ Command failed with exit code: ${exitCode}`);
            if (stderr) {
              logs.push(`Error: ${stderr}`);
            }
            reject(new Error(`Command failed: ${stderr}`));
          }
        });
        
        stream.on('data', (data: Buffer) => {
          stdout += data.toString();
        });
        
        stream.stderr.on('data', (data: Buffer) => {
          stderr += data.toString();
        });
      });
    });
    
    // Check for Triton-Puzzles directory
    logs.push('');
    logs.push('=== Checking for Triton-Puzzles directory ===');
    
    if (commandOutput.includes('Triton-Puzzles')) {
      logs.push('✅ Triton-Puzzles directory found');
    } else {
      logs.push('⚠️  Triton-Puzzles directory not found');
    }
    
    conn.end();
    
    logs.push('');
    logs.push('=== Test Complete ===');
    logs.push('✅ SSH connection is working correctly!');
    
    console.log('SSH test completed successfully');
    res.json({
      success: true,
      message: 'SSH connection test successful',
      logs: logs.join('\n'),
    });
    
  } catch (error) {
    conn.end();
    logs.push('');
    logs.push('=== Test Failed ===');
    logs.push(`❌ Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    
    console.error('SSH test failed:', error);
    res.json({
      success: false,
      message: 'SSH connection test failed',
      logs: logs.join('\n'),
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

// Health check endpoint
app.get('/health', (_req: Request, res: Response) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.listen(PORT, () => {
  console.log(`\n🚀 Server running on http://localhost:${PORT}`);
  console.log(`📡 Translation endpoint: POST /api/translate`);
  console.log(`🔍 Verification endpoint: POST /api/verify`);
  console.log(`🧪 Test SSH endpoint: POST /api/test-ssh`);
  console.log(`💚 Health check: GET /health\n`);
});

export default app;

