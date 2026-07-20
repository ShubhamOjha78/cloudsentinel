#!/bin/bash
# ============================================================
# CloudSentinel – setup.sh
# Auto-install script for Ubuntu 22.04 LTS (AWS EC2)
# Usage: bash setup.sh
# ============================================================

set -e   # Exit on any error

echo "=================================================="
echo " CloudSentinel – Installation Script"
echo " MCA (AI/ML) Project | Chandigarh University"
echo "=================================================="

# 1. System updates
echo "[1/7] Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y python3 python3-pip python3-venv postgresql postgresql-contrib git -qq

# 2. Python virtual environment
echo "[2/7] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
echo "[3/7] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# 4. PostgreSQL setup
echo "[4/7] Setting up PostgreSQL database..."
sudo service postgresql start
sudo -u postgres psql -c "CREATE DATABASE cloudsentinel;" 2>/dev/null || echo "DB may already exist — continuing."
sudo -u postgres psql -c "CREATE USER clouduser WITH PASSWORD 'cloudpass';" 2>/dev/null || true
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE cloudsentinel TO clouduser;" 2>/dev/null || true
sudo -u postgres psql -d cloudsentinel -f database/schema.sql

# 5. Environment config
echo "[5/7] Setting up environment config..."
if [ ! -f config.env ]; then
    cp config.env.example config.env
    echo "  ⚠️  Created config.env from example — please edit it with your credentials!"
else
    echo "  config.env already exists — skipping."
fi

# 6. Create model save directory
echo "[6/7] Creating model directory..."
mkdir -p models/saved

# 7. Final message
echo "[7/7] Setup complete!"
echo ""
echo "=================================================="
echo " Next Steps:"
echo "  1. Edit config.env with your DB and SMTP details"
echo "  2. Collect data: python agents/collector.py --instance-id YOUR_INSTANCE_ID"
echo "  3. Train models (after 30 min of data): python train_all.py --instance-id YOUR_INSTANCE_ID"
echo "  4. Start detection: python detection_loop.py --instance-id YOUR_INSTANCE_ID"
echo "  5. Launch dashboard: streamlit run dashboard/app.py --server.port 8501"
echo "=================================================="
