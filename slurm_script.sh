#!/bin/bash

for RBP in $(ls datasets); do
	cat << __EOF__ > train_$RBP.sl
#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p volta-gpu
#SBATCH --mem=4g
#SBATCH -t 20:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:4

/bin/bash /proj/magnuslb/users/mkratz/bert-rbp/scripts/train_and_test.sh $RBP
__EOF__

	JOBID=$(sbatch train_$RBP.sl | cut -d' ' -f4)
	DATE=$(date "+%Y-%m-%d %H:%M:%S")
	echo "$DATE	$RBP	$JOBID" >> trainings.tsv	
done
