include Configuration
SBD_INCLUDE_DIR=$(SBD_PATH)/include
# Add library path for BLAS/LAPACK
# Set BLAS_LIBDIR environment variable to override (e.g., export BLAS_LIBDIR=/custom/blas/path)
# If BLAS/LAPACK libraries are found automatically by the compiler, you can omit LIBPATH
LIBPATH=$(if $(BLAS_LIBDIR),-L$(BLAS_LIBDIR),)$(if $(wildcard $(HOME)/lib/libblas.so),-L$(HOME)/lib,)
LIBFLAGS= $(LIBPATH) $(SYSLIB)
# Allow additional flags from command line
EXTRA_CFLAGS=
MAKEFILES= Makefile Configuration
# header file
HEADER=
# source
SOURCES= main.cc
#objects
OBJECTS=
# compilation
.SUFFIXES:
.SUFFIXES: .h .cc .o
.cc.o:  $*.cc
	$(CCCOM) -c $(CCFLAGS) $(EXTRA_CFLAGS) -I$(SBD_INCLUDE_DIR) $<
###############################################################
diag: clean main.o $(OBJECTS)
	$(CCCOM) $(CCFLAGS) $(EXTRA_CFLAGS) -o diag main.o $(OBJECTS) $(LIBFLAGS)

clean:
	rm -f main.o

###############################################################
