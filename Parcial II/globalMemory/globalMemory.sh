#!/bin/bash

#SBATCH --job-name=globalMemory
#SBATCH --output=globalMemory.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=0

./globalMemory ../Images/bentley.jpg
./globalMemory ../Images/bugatti.jpg
./globalMemory ../Images/chevrolet.jpg
./globalMemory ../Images/ferrari.jpg
./globalMemory ../Images/ford.jpg
./globalMemory ../Images/koenigsegg.jpg
./globalMemory ../Images/lamborghini.jpg
./globalMemory ../Images/mercedes.jpg
./globalMemory ../Images/nissan.jpg
./globalMemory ../Images/pagani.jpg
