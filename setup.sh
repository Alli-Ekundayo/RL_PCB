#!/bin/bash

CUDA_VERSION=12.4

# Setup repository environment variable
RL_PCB=${PWD}
export RL_PCB
echo "RL_PCB=${RL_PCB}"

# Make CUDA available
if [ -d "/usr/local/cuda-${CUDA_VERSION}" ]; then
	export PATH="/usr/local/cuda-${CUDA_VERSION}/bin:$PATH"
	export LD_LIBRARY_PATH="/usr/local/cuda-${CUDA_VERSION}/lib64:$LD_LIBRARY_PATH"
elif [ -d "/usr/local/cuda" ]; then
	export PATH="/usr/local/cuda/bin:$PATH"
	export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
	echo "Using default CUDA installation at /usr/local/cuda"
elif command -v nvcc &>/dev/null; then
	# CUDA installed via apt (nvidia-cuda-toolkit) - nvcc in /usr/bin, libs in /usr/lib
	NVCC_PATH=$(dirname $(which nvcc))
	export PATH="${NVCC_PATH}:$PATH"
	# System CUDA libs are needed for nvcc, but PyPI CUDA packages have their own libs
	# We'll set this but ensure venv libs take precedence (see after venv activation)
	export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
	echo "Using apt-installed CUDA toolkit (nvcc at $(which nvcc))"
else
	echo "Could not find CUDA ${CUDA_VERSION} on system. GPU support may not be available."
fi

echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "PATH=${PATH}"

# Activate virtual environment
echo -n "Attempting to activate virtual environment ${RL_PCB}/venv ... "
if [ -d "${RL_PCB}/venv" ]; then
	source ${RL_PCB}/venv/bin/activate	# True virutal environment
	echo "OK"
	# Prepend venv lib paths to ensure PyPI CUDA packages take precedence over system CUDA
	# This fixes version mismatches (e.g., PyPI CUDA 12.1 vs system CUDA 12.0)
	if [ -d "${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia" ]; then
		export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/cublas/lib:${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/cudnn/lib:${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/cufft/lib:${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/curand/lib:${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/cusolver/lib:${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/nccl/lib:${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/nvtx/lib:$LD_LIBRARY_PATH"
		echo "Prioritized PyPI CUDA libraries over system CUDA"
	fi
else
	echo "Failed, venv does not exist."
	echo "Please use script 'create_venv.sh' to automatically setup the environment."
fi
