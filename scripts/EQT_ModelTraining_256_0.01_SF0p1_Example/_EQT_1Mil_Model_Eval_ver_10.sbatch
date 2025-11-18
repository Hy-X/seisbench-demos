#!/bin/bash
#
#SBATCH --partition=disc
#SBATCH --gres=gpu:1
#SBATCH --output=EQT_1Mil_%J_.txt
#SBATCH --error=EQT_1Mil_%J_.txt
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --mail-user=hongyu.xiao-1@ou.edu      
#SBATCH --mail-type=BEGIN,END,FAIL             

python EQT_1mil_120s_Eval_Hist_ver_10.py
