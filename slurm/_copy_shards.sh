# Source this file from SLURM training scripts to copy shards to $SCRATCH.
# Usage:  source slurm/_copy_shards.sh
#
# Sets ASL_SHARDS_DIR to the fast local copy.
# Falls back to $HOME/asl_shards if $DATA/asl_shards is not found.
# Aborts the parent job if shards cannot be located anywhere.

# Prefer $DATA (large-file storage), fall back to $HOME
if [ -d "$DATA/asl_shards" ]; then
    SRC="$DATA/asl_shards"
elif [ -d "$HOME/asl_shards" ]; then
    echo "[shards] WARNING: $DATA/asl_shards not found, using $HOME/asl_shards"
    SRC="$HOME/asl_shards"
else
    echo "[shards] ERROR: asl_shards not found in \$DATA or \$HOME. Aborting." >&2
    exit 1
fi

# $SCRATCH on Pirineus 3 is already per-job; use task ID to avoid race conditions
# when multiple array tasks land on the same node.
DST="$SCRATCH/asl_shards_${SLURM_ARRAY_TASK_ID:-0}"

mkdir -p "$DST"
echo "[shards] Copying $SRC -> $DST ..."
cp -r "$SRC/." "$DST/"

if [ $? -ne 0 ]; then
    echo "[shards] ERROR: cp failed. Aborting." >&2
    exit 1
fi

echo "[shards] Copy done. $(du -sh "$DST" | cut -f1) on scratch."
export ASL_SHARDS_DIR="$DST"
