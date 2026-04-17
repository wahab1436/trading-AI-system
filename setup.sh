#!/bin/bash

# Trading AI System - Setup Script

set -e

echo "🚀 Trading AI System Setup"
echo "=========================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Setup DVC
echo "💾 Setting up DVC..."
dvc init --no-scm
dvc remote add -d storage s3://trading-ai-data/

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p data/{raw,processed,labels,images,metadata}
mkdir -p models/{cnn,fusion,champion,challenger,retired}
mkdir -p logs
mkdir -p notebooks
mkdir -p tests

# Copy environment template
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠️  Please edit .env file with your credentials"
fi

# Setup pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
echo "Running pre-commit checks..."
black --check .
flake8 .
mypy .
EOF
chmod +x .git/hooks/pre-commit

# Initialize MLflow
echo "📊 Initializing MLflow..."
mkdir -p mlflow_artifacts
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow_artifacts --host 0.0.0.0 --port 5000 &
MLFLOW_PID=$!

# Wait for MLflow to start
sleep 5
kill $MLFLOW_PID 2>/dev/null || true

# Create first experiment
python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.create_experiment('trading-ai-fusion')
" 2>/dev/null || true

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your broker credentials"
echo "2. Run: python data_engine/historical_fetch.py --years 5"
echo "3. Run: python etl/flows/label_data.py"
echo "4. Run: python etl/flows/render_images.py"
echo "5. Run: python cnn_model/train.py"
echo "6. Run: python fusion_model/train_xgb.py"
echo ""
echo "Start paper trading: python main.py --mode paper"
echo "Start API server: python api/main.py"
echo ""
