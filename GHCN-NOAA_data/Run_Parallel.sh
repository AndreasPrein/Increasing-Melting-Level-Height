#!/bin/bash
#
# This script is the looper for annual GHCN data
# preprocessing
#
###########################################
#                        imput section
###########################################

iStartYear=1979
iStopYear=2016

# loop over the years
for yy in $(seq $iStartYear $iStopYear)
do
    if [ ! -f '/glade/scratch/prein/Papers/Trends_RadSoundings/data/GHCN-NOAA/ghcnd_all/npz/'$yy'_GHCN.npz' ]; then
	echo ' process year '$yy 

	# prepare the parallel python script
	PythonName=$yy'_Process_GHCN-NOAA_data.py'
	sed "s#>>YYYY<<#$yy#g" 'Process_GHCN-NOAA_data.py' > $PythonName
	#sed -i "s#>>TC_STOP<<#$iTCstop#g" $PythonName
	chmod 744 $PythonName

cat <<EOF >>$yy'_BATCH.sh'
#!/bin/bash -l
#SBATCH -J $yy
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200G
#SBATCH -t 24:00:00
#SBATCH -A P66770001
#SBATCH -p dav

module load python/2.7.14
ncar_pylib
ml ncl nco
srun ./$PythonName
EOF
chmod 744 $yy'_BATCH.sh'
sbatch $yy'_BATCH.sh'

        sleep 2

	# Cheyenne Submission
	# ml ncl nco
	# # qsub -l select=1:ncpus=36:mpiprocs=36:mem=109GB -l walltime=12:00:00  -q economy -A P66770001 ./$PythonName
	# qsub -l select=1:ncpus=36:mpiprocs=36:mem=109GB -l walltime=12:00:00  -q economy -A NMMM0035 ./$PythonName
        
	# squeue -u $USER
    fi
done

exit

