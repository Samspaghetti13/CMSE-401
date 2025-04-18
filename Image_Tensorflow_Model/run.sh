#!/bin/bash
#SBATCH --job-name=chest_xray_classification
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --error=logs/job_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

module purge
module load GCC
module load CUDA
module load OpenMPI
module load Python
module load TensorFlow

mkdir -p logs models results tf_cache

if [ ! -d "chest_xray_env" ]; then
  echo "Creating virtual environment..."
  python -m venv chest_xray_env
  source chest_xray_env/bin/activate

  pip install --upgrade pip
  pip install numpy pandas scikit-learn==1.2.2 tensorflow tensorflow-addons opencv-python matplotlib seaborn tqdm pillow h5py

  pip freeze > requirements_used.txt
else
  echo "Using existing virtual environment..."
  source chest_xray_env/bin/activate
fi

export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_CACHE_DISABLE=0
export TF_GPU_THREAD_MODE=gpu_private
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices --tf_xla_auto_jit=2"

echo "Starting training process..."
start_time=$(date +%s)

python model.py 2>&1 | tee logs/training_log_$(date + "%Y%m%d_%H%M%S").txt

if [ $? -eq 0 ]; then
  echo "Training completed successfully!"
else
  echo "Training failed with exit code $?"
  exit 1
fi

end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

echo "Job completed at: $(date)"
echo "Total execution time: ${hours}h ${minutes}m ${seconds}s"

deactivate

echo "Job Finished!"
