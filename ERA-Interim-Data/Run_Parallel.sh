#!/bin/bash
#
# This script is the looper for monthly 3D Storm analyses
#
# To monitor jobs use: qstat -xu $USER
#
###########################################
#                        imput section
###########################################

iStartYear=1979
iStopYear=2017

# loop over the years
for yy in $(seq $iStartYear $iStopYear)
do
    if [ ! -f '/glade/scratch/prein/Papers/Trends_RadSoundings/data/ERA-Int/'$yy'121600_'$yy'123121_ERA-Int_Hail-Env_RDA.nc' ]; then
	echo ' process year '$yy 

	# prepare the parallel python script
	PythonName=$yy'_ERA-Interim-Data-processing.py'
	sed "s#>>YYYY<<#$yy#g" 'ERA-Interim-Data-processing.py' > $PythonName
	# sed -i "s#>>TC_STOP<<#$iTCstop#g" $PythonName
	chmod 744 $PythonName

	ml ncl
	# qsub -l select=1:ncpus=36:mpiprocs=36:mem=109GB -l walltime=12:00:00  -q economy -A P66770001 ./$PythonName
	qsub -l select=1:ncpus=1:mpiprocs=1:mem=109GB -l walltime=12:00:00  -q economy -A NMMM0035 ./$PythonName

	sleep 5
    fi
done

exit

