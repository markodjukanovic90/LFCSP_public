TARGET = ILP_LFCS
CXXFLAGS = -ansi -O3 -std=c++17
GREOBJS = utils.o UniformSampling.o Instance.o

SYSTEM     = x86-64_linux
LIBFORMAT  = static_pic
CPLEXDIR      =  /home/marko/cplex_2025/cplex
CONCERTDIR    =  /home/marko/cplex_2025/concert
CCC = g++
CCOPT = -m64 -O -fPIC -fexceptions -DNDEBUG -DIL_STD -std=c++11 -fpermissive -w -mcmodel=medium 
CPLEXBINDIR   = $(CPLEXDIR)/bin/$(BINDIST)
CPLEXLIBDIR   = $(CPLEXDIR)/lib/$(SYSTEM)/$(LIBFORMAT)
CONCERTLIBDIR = $(CONCERTDIR)/lib/$(SYSTEM)/$(LIBFORMAT)
CCLNFLAGS = -L$(CPLEXLIBDIR) -lilocplex -lcplex -L$(CONCERTLIBDIR) -lconcert -lm -pthread
CLNFLAGS  = -L$(CPLEXLIBDIR) -lcplex -lm -pthread
CONCERTINCDIR = $(CONCERTDIR)/include
CPLEXINCDIR   = $(CPLEXDIR)/include
CCFLAGS = $(CCOPT) -I$(CPLEXINCDIR) -I$(CONCERTINCDIR)


all: ${TARGET}

ILP_LFCS: ILP_LFCS.o $(GREOBJS)
	$(CCC) $(CXXFLAGS) $(CCFLAGS)  ILP_LFCS.o $(GREOBJS) -o ILP_LFCS $(CCLNFLAGS)

ILP_LFCS.o: ILP_LFCS.cpp
	$(CCC) $(CXXFLAGS) -c $(CCFLAGS) ILP_LFCS.cpp -o ILP_LFCS.o 

clean:
	@rm -f *~ *.o ${TARGET} corex86-64_linux
