#!/bin/bash

echo "=========================================="
echo "GPU Kernel Translator - Setup Check"
echo "=========================================="
echo ""

# Check Node.js
echo "🔍 Checking Node.js..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo "✅ Node.js installed: $NODE_VERSION"
else
    echo "❌ Node.js not found. Please install Node.js 18+"
    exit 1
fi

echo ""

# Check Ollama
echo "🔍 Checking Ollama..."
if command -v ollama &> /dev/null; then
    echo "✅ Ollama CLI installed"
    
    # Check if Ollama server is running
    if curl -s http://localhost:11434/api/version &> /dev/null; then
        echo "✅ Ollama server is running"
        
        # Check if gemma3n:latest model is available
        if ollama list | grep -q "gemma3n:latest"; then
            echo "✅ gemma3n:latest model is available"
        else
            echo "⚠️  gemma3n:latest model not found"
            echo "   Run: ollama pull gemma3n:latest"
        fi
    else
        echo "⚠️  Ollama server is not running"
        echo "   Run: ollama serve"
    fi
else
    echo "❌ Ollama not found. Please install from https://ollama.ai"
fi

echo ""

# Check SSH key
echo "🔍 Checking SSH configuration..."
SSH_KEY="$HOME/.ssh/id_ed25519"
if [ -f "$SSH_KEY" ]; then
    echo "✅ SSH key found at $SSH_KEY"
    
    # Check SSH key permissions
    PERMS=$(stat -f "%Lp" "$SSH_KEY" 2>/dev/null || stat -c "%a" "$SSH_KEY" 2>/dev/null)
    if [ "$PERMS" = "600" ] || [ "$PERMS" = "400" ]; then
        echo "✅ SSH key has correct permissions"
    else
        echo "⚠️  SSH key permissions should be 600"
        echo "   Run: chmod 600 $SSH_KEY"
    fi
    
    # Test SSH connection (with timeout)
    echo "🔍 Testing SSH connection to root@134.199.201.182..."
    if timeout 5 ssh -i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no root@134.199.201.182 "echo 'Connection successful'" 2>/dev/null | grep -q "Connection successful"; then
        echo "✅ SSH connection successful"
    else
        echo "⚠️  Could not connect to remote server"
        echo "   This might be normal if the server is not accessible from your network"
    fi
else
    echo "❌ SSH key not found at $SSH_KEY"
    echo "   Make sure you have the correct SSH key"
fi

echo ""

# Check if dependencies are installed
echo "🔍 Checking npm dependencies..."
if [ -d "node_modules" ]; then
    echo "✅ node_modules directory exists"
else
    echo "⚠️  Dependencies not installed"
    echo "   Run: npm install"
fi

echo ""
echo "=========================================="
echo "Setup check complete!"
echo "=========================================="
echo ""
echo "To start the application:"
echo "  npm run dev:all    (starts both frontend and backend)"
echo ""
echo "Or separately:"
echo "  npm run dev        (frontend only)"
echo "  npm run server     (backend only)"
echo ""

