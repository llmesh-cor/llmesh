#!/bin/bash

# LLMESH Network Setup Script

echo "🚀 LLMESH Network Setup"
echo "===================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)"; then
    echo "❌ Python 3.8+ is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Install package
echo "📦 Installing MESH package..."
pip install -e .

# Create directories
echo "📁 Creating directories..."
mkdir -p data logs models

# Generate node ID
node_id=$(python3 -c "import hashlib, uuid; print(hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()[:16])")
echo "🔑 Generated Node ID: $node_id"

# Create config file
echo "📝 Creating configuration..."
cp mesh.config.yaml mesh.config.local.yaml
sed -i "s/id: null/id: $node_id/" mesh.config.local.yaml

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Edit configuration: nano mesh.config.local.yaml"
echo "3. Run node: python examples/node_setup.py"
echo ""
echo "For more information, see docs/README.md"
