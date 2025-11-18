#!/bin/bash
#
#SBATCH --partition=ogsgpu_el9_tmp
#SBATCH --gres=gpu:1
#SBATCH --output=PNet_1Mil_%J_.txt
#SBATCH --error=PNet_1Mil_%J_.err

#SBATCH --mem=20G
#SBATCH --time=48:00:00

#SBATCH --mail-user=hongyu.xiao-1@ou.edu      
#SBATCH --mail-type=BEGIN,END,FAIL             

python PhaseNet_Eval_ExampleAndHist_ver_4.py
