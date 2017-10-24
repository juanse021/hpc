#!/bin/bash

#SBATCH --job-name=sharedMemory
#SBATCH --output=sharedMemory.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=0

./sharedMemory ../Images/bentley.jpg 20
./sharedMemory ../Images/bugatti.jpg 20
./sharedMemory ../Images/chevrolet.jpg 20
./sharedMemory ../Images/ferrari.jpg 20
./sharedMemory ../Images/ford.jpg 20
./sharedMemory ../Images/koenigsegg.jpg 20
./sharedMemory ../Images/lamborghini.jpg 20
./sharedMemory ../Images/maserati.jpg 20
./sharedMemory ../Images/nissan.jpg 20
./sharedMemory ../Images/pagani.jpg 20
