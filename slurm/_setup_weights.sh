# Source this file from SLURM transfer-learning scripts to make torchvision
# find pretrained weights at $DATA/pretrained_weights/*.pth without internet.
# Usage:  source slurm/_setup_weights.sh
#
# Expects flat .pth files uploaded to $DATA/pretrained_weights/
# Aborts the parent job if no weights are found.

WEIGHTS_DIR="$DATA/pretrained_weights"

if [ -z "$(ls -A "$WEIGHTS_DIR"/*.pth 2>/dev/null)" ]; then
    echo "[weights] ERROR: no .pth files found in $WEIGHTS_DIR" >&2
    echo "[weights]   Upload pretrained_weights/*.pth to \$DATA/pretrained_weights/" >&2
    exit 1
fi

# Build a thin TORCH_HOME wrapper: hub/checkpoints/ -> WEIGHTS_DIR
TORCH_HOME_DIR="$DATA/.torch_home"
mkdir -p "$TORCH_HOME_DIR/hub"
if [ ! -L "$TORCH_HOME_DIR/hub/checkpoints" ]; then
    ln -s "$WEIGHTS_DIR" "$TORCH_HOME_DIR/hub/checkpoints"
fi

echo "[weights] Found weights in $WEIGHTS_DIR"
export TORCH_HOME="$TORCH_HOME_DIR"
echo "[weights] TORCH_HOME -> $TORCH_HOME"
