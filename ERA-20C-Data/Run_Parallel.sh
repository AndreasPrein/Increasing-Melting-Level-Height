#!/bin/bash
#
# This script is the looper for monthly 3D Storm analyses
#
# Check status of jobs with: qstat -xu $USER
#
###########################################
#                        imput section
###########################################

iStartYear=1900
iStopYear=1978

# loop over the years
for yy in $(seq $iStartYear $iStopYear)
do
    if [ ! -f '/glade/scratch/prein/Papers/Trends_RadSoundings/data/ERA-20c/Full/'$yy'121600_'$yy'123121_ERA-20c_Hail-Env_RDA.nc' ]; then
	echo ' process year '$yy

	# prepare the parallel python script
	PythonName=$yy'_ERA-20C-Data-processing.py'
	sed "s#>>YYYY<<#$yy#g" 'ERA-20C-Data-processing.py' > $PythonName
	# sed -i "s#>>TC_STOP<<#$iTCstop#g" $PythonName
	chmod 744 $PythonName

	ml ncl nco
	# qsub -l select=1:ncpus=36:mpiprocs=36:mem=109GB -l walltime=12:00:00  -q economy -A P66770001 ./$PythonName
	qsub -l select=1:ncpus=1:mpiprocs=1:mem=109GB -l walltime=12:00:00  -q economy -A NMMM0035 ./$PythonName
            
	sleep 1
    fi
done

exit

