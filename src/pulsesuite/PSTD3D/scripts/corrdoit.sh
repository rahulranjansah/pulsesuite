############################################################################
# A file to configure and compile 2DPSTD  on linux servers.
# Set the different options to the desired values and execute
###########################################################################

##  SET THE PATH TO THE PROJECT DIRECTORY
path0=$HOME/Desktop/SFFP/propagation


##  SET THE PATH TO SOURCE DIRECTORIES IN LIBPULSESUITE AND PULSESUITE
p0=$path0/pulsesuite/libpulsesuite/src
p1=$path0/pulsesuite/src


## SET THE PATH TO PULSESUITE INSTALLATION DIRECTORIES
p2=$path0/local/include
p3=$path0/local/lib


##  SET THE PATH TO FFTW3 INSTALLATION DIRECTORY
fftw3=$path0/fftw3


#  SET COMPILER FLAGS FOR IFORT (COMMENT OUT IF NOT USING IFORT) 
FC=ifort # FOR IFORT
OMP='-openmp -O2'  # FOR IFORT


## SET COMPILER FLAGS FOR GFORTRAN (COMMENT OUT IF NOT USING GFORTRAN)
#FC=gfortran-8        # FOR GFORTRAN
#OMP='-fopenmp'     # FOR GFORTRAN


##  COPY NEEDED ITEMS FROM PULSESUITE PACKAGE TO 'STUFF' FOLDER 
cd stuff
./doit.sh
cd ..


##  ENTER THE 2DPSTD SOURCE DIRECTORY
mkdir corr
cd corr
cp ../src/correlations.f90 .



$FC $OMP -I$p1 -I$p2 -I$p0 -c -o   correlations.o       correlations.f90
$FC $OMP -I$p1 -I$p2 -L$p3 -L$fftw3/lib -o correlations  *.o ../stuff/*.o ../stuff/libpropagator.a ../stuff/libtalanov.a ../stuff/libsplitstep.a ../stuff/libpulseproperties.a ../stuff/libpulsesuite.a -lfftw3
mv correlations ..



##  EXIT 2DPSTD SOURCE DIRECTORY
cd ..



