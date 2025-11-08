#!/usr/bin/env node

/**
 * Simple SSH Connection Test
 * Tests SSH connection to remote server and executes ls command
 */

const { Client } = require('ssh2');
const fs = require('fs');
const path = require('path');
const os = require('os');

console.log('==========================================');
console.log('SSH Connection Test');
console.log('==========================================\n');

const sshKeyPath = path.join(os.homedir(), '.ssh', 'id_ed25519');
const host = '134.199.201.182';
const username = 'root';

// Check if SSH key exists
if (!fs.existsSync(sshKeyPath)) {
  console.error('❌ SSH key not found at:', sshKeyPath);
  process.exit(1);
}

console.log('✓ SSH key found:', sshKeyPath);
console.log('✓ Connecting to:', `${username}@${host}`);
console.log('');

const conn = new Client();
const logs = [];

conn.on('ready', () => {
  logs.push('✅ SSH connection established successfully!');
  console.log('✅ SSH connection established successfully!\n');
  
  // Execute ls command
  console.log('Executing command: ls -la\n');
  logs.push('Executing command: ls -la');
  
  conn.exec('ls -la', (err, stream) => {
    if (err) {
      console.error('❌ Command execution failed:', err.message);
      logs.push(`❌ Command execution failed: ${err.message}`);
      conn.end();
      process.exit(1);
    }

    let stdout = '';
    let stderr = '';

    stream.on('close', (code, signal) => {
      logs.push(`Command completed with exit code: ${code}`);
      
      console.log('==========================================');
      console.log('Command Output (stdout):');
      console.log('==========================================');
      console.log(stdout || '(no output)');
      
      if (stderr) {
        console.log('\n==========================================');
        console.log('Errors (stderr):');
        console.log('==========================================');
        console.log(stderr);
      }
      
      console.log('\n==========================================');
      console.log('Summary:');
      console.log('==========================================');
      logs.forEach(log => console.log(log));
      
      if (code === 0) {
        console.log('\n✅ SUCCESS: SSH connection and command execution successful!');
      } else {
        console.log(`\n⚠️  Command exited with code: ${code}`);
      }
      
      conn.end();
    });

    stream.on('data', (data) => {
      stdout += data.toString();
    });

    stream.stderr.on('data', (data) => {
      stderr += data.toString();
    });
  });
});

conn.on('error', (err) => {
  console.error('❌ SSH connection error:', err.message);
  logs.push(`❌ SSH connection error: ${err.message}`);
  
  console.log('\nTroubleshooting:');
  console.log('1. Check if SSH key has correct permissions: chmod 600', sshKeyPath);
  console.log('2. Test manual connection: ssh -i', sshKeyPath, `${username}@${host}`);
  console.log('3. Check if you can reach the server: ping', host);
  
  process.exit(1);
});

// Load SSH private key
const privateKey = fs.readFileSync(sshKeyPath);

// Connect
console.log('Connecting...\n');
conn.connect({
  host: host,
  port: 22,
  username: username,
  privateKey: privateKey,
  readyTimeout: 30000,
});

