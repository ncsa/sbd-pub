include Configuration
SBD_INCLUDE_DIR=$(SBD_PATH)/include
LIBFLAGS= $(SYSLIB)
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
	$(CCCOM) -c $(CCFLAGS) -I$(SBD_INCLUDE_DIR) $<
###############################################################
gdet: clean main.o $(OBJECTS)
	$(CCCOM) $(CCFLAGS) -o gdet main.o $(OBJECTS) $(LIBFLAGS)

clean:
	rm -f main.o

###############################################################
