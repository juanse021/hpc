#!/bin/bash

#SBATCH --job-name=globalMemory
#SBATCH --output=globalMemory.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=0

./globalMemory ../Images/bentley.jpg 20
./globalMemory ../Images/bugatti.jpg 20
./globalMemory ../Images/chevrolet.jpg 20
./globalMemory ../Images/ferrari.jpg 20
./globalMemory ../Images/ford.jpg 20
./globalMemory ../Images/koenigsegg.jpg 20
./globalMemory ../Images/lamborghini.jpg 20
./globalMemory ../Images/maserati.jpg 20
./globalMemory ../Images/nissan.jpg 20
./globalMemory ../Images/pagani.jpg 20