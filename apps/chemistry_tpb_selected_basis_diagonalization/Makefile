include Configuration
SBD_INCLUDE_DIR=$(SBD_PATH)/include
# Add library path for BLAS/LAPACK
# Set BLAS_LIBDIR environment variable to override (e.g., export BLAS_LIBDIR=/custom/blas/path)
LIBFLAGS= $(LDFLAGS) $(SYSLIB)
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
	$(CCCOM) -o diag main.o $(OBJECTS) $(LIBFLAGS)

clean:
	rm -f main.o

###############################################################
