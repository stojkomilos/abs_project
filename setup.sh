#!/usr/bin/env bash
# Run once on the remote GPU instance to set up everything.
# Usage: bash setup.sh <kaggle_username> <kaggle_api_key>
#
# Get your API key at: https://www.kaggle.com/settings -> API -> Create New Token

set -euo pipefail

KAGGLE_USERNAME=${1:?usage: bash setup.sh <kaggle_username> <kaggle_api_key>}
KAGGLE_KEY=${2:?usage: bash setup.sh <kaggle_username> <kaggle_api_key>}

echo "=== 1. System packages ==="
apt-get update -qq && apt-get install -y -qq unzip curl git

echo "=== 2. Python packages ==="
pip install -q torch torchvision torchaudio \
    scikit-learn matplotlib jupyter notebook tqdm kaggle

echo "=== 3. Kaggle credentials ==="
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json <<EOF
{"username":"${KAGGLE_USERNAME}","key":"${KAGGLE_KEY}"}
EOF
chmod 600 ~/.kaggle/kaggle.json

echo "=== 4. Download dataset ==="
mkdir -p dataset
kaggle datasets download -d abdullahkhan70/lumbar-spinal-mri-dataset -p dataset --unzip
echo "Dataset contents:"
ls dataset/

echo "=== 5. Clone/copy project files ==="
# If you have a git repo, replace this with: git clone <your_repo_url>
# Otherwise the notebook + train.py are downloaded below as a fallback.
if [ ! -f lumbar_spine_cnn.ipynb ]; then
    echo "WARNING: lumbar_spine_cnn.ipynb not found."
    echo "Either git clone your repo here, or scp the files manually:"
    echo "  scp -P <port> lumbar_spine_cnn.ipynb train.py root@<host>:~/"
fi

echo ""
echo "=== Done! ==="
echo "Start Jupyter with:"
echo "  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
echo "Then open the URL shown in the output in your browser (or tunnel via VSCode Remote SSH)."
