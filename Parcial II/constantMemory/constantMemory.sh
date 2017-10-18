#!/bin/bash

#SBATCH --job-name=constantMemory
#SBATCH --output=constantMemory.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=0

./constantMemory ../Images/bentley.jpg
./constantMemory ../Images/bugatti.jpg
./constantMemory ../Images/chevrolet.jpg
./constantMemory ../Images/ferrari.jpg
./constantMemory ../Images/ford.jpg
./constantMemory ../Images/koenigsegg.jpg
./constantMemory ../Images/lamborghini.jpg
./constantMemory ../Images/maserati.jpg
./constantMemory ../Images/nissan.jpg
./constantMemory ../Images/pagani.jpg
