#!/bin/bash

#SBATCH --account=tbag175
#SBATCH -D /arf/scratch/tbag175
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=14:00:00
#SBATCH --job-name=test
#SBATCH --output /arf/home/tbag175/slurm_output.txt

#SBATCH --partition=palamut-cuda
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

#SBATCH --requeue
# Check if the job has been restarted
if [[ -z "$SLURM_RESTART_COUNT" ]]; then
    export SLURM_RESTART_COUNT=0
else
    export SLURM_RESTART_COUNT=$((SLURM_RESTART_COUNT + 1))
fi

### Load modules

echo "We have the modules: $(module list 2>&1)" > ${SLURM_JOB_ID}.info

echo "SLURM_NODELIST $SLURM_NODELIST"
echo "NUMBER OF CORES $SLURM_NTASKS"

wdir=/arf/home/tbag175   #Uygulama için çalıştırılacak dosyaların adresleri.
cd $wdir

infile=$wdir/comment_gatherer.py
infile2=$wdir/progress.json #Uygulamanın çalışması için gerekli input ve output dosyaların adları.
outfile=$wdir/youtube_comments.csv
outfile2=$wdir/progress.json

### jobs
source $wdir/env/bin/activate
python comment_gatherer.py

# Check if the maximum retries have been reached
if [[ $SLURM_RESTART_COUNT -ge 5 ]]; then
    echo "Maximum retries reached. Exiting."
    exit 1
fi

exit

