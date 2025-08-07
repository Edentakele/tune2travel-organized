#!/bin/bash
#SBATCH -p mercan
#SBATCH -A sefa
#SBATCH -J gaussian09_test
#SBATCH -N 1 # hesabin dagitilacagi node adedi
#SBATCH -n 4 # is icin toplamda kullanilacak cekirdek adeti
#SBATCH --time=2:00:00
export g09root=$HOME
export GAUSS_SCRDIR=/tmp/$SLURM_JOB_ID
mkdir -p $GAUSS_SCRDIR
. $g09root/g09/bsd/g09.profile
echo "SLURM_NODELIST $SLURM_NODELIST"
$g09root/g09/g09 < gaussian_egitim.com
rm -rf $GAUSS_SCRDIR
exit
