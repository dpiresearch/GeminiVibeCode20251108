#!/bin/bash

echo "=========================================="
echo "Testing SSH Connection to Remote Server"
echo "=========================================="
echo ""

# Check if backend server is running
echo "🔍 Checking if backend server is running..."
if curl -s http://localhost:3001/health &> /dev/null; then
    echo "✅ Backend server is running"
else
    echo "❌ Backend server is not running"
    echo "   Please start it with: npm run server"
    exit 1
fi

echo ""
echo "🔍 Testing SSH connection with ls command..."
echo ""

# Create a simple test that will run ls command
TEST_CODE='import subprocess
import sys

try:
    result = subprocess.run(["ls", "-la"], capture_output=True, text=True)
    print("Directory listing:")
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr, file=sys.stderr)
    sys.exit(result.returncode)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
'

# Make API call to verify endpoint
RESPONSE=$(curl -s -X POST http://localhost:3001/api/verify \
  -H "Content-Type: application/json" \
  -d "{
    \"language\": \"Triton\",
    \"code\": $(echo "$TEST_CODE" | jq -Rs .)
  }")

echo "=========================================="
echo "Response from server:"
echo "=========================================="
echo "$RESPONSE" | jq -r '.result' || echo "$RESPONSE"
echo ""
echo "=========================================="
echo ""

# Check if successful
if echo "$RESPONSE" | grep -q "COMPILATION STATUS: SUCCESS"; then
    echo "✅ SSH connection and command execution successful!"
else
    echo "⚠️  Check the response above for details"
fi

