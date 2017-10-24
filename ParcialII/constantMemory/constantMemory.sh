#!/bin/bash

#SBATCH --job-name=constantMemory
#SBATCH --output=constantMemory.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=0

./constantMemory ../Images/bentley.jpg 20
./constantMemory ../Images/bugatti.jpg 20
./constantMemory ../Images/chevrolet.jpg 20
./constantMemory ../Images/ferrari.jpg 20
./constantMemory ../Images/ford.jpg 20
./constantMemory ../Images/koenigsegg.jpg 20
./constantMemory ../Images/lamborghini.jpg 20
./constantMemory ../Images/maserati.jpg 20
./constantMemory ../Images/nissan.jpg 20
./constantMemory ../Images/pagani.jpg 20
