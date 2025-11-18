#!/bin/bash
#
#SBATCH --partition=disc
#SBATCH --gres=gpu:1
#SBATCH --output=GPD_1Mil_%J_.txt
#SBATCH --error=GPD_1Mil_%J_.err
#SBATCH --mem=30G
#SBATCH --time=48:00:00
#SBATCH --mail-user=hongyu.xiao-1@ou.edu      
#SBATCH --mail-type=BEGIN,END,FAIL             

python GPD_Local_Enhance_Dual_ver_3.py
