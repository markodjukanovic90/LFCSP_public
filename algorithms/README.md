## Compilation and Execution Guide

### Compiling the Sources

1. **ILP_LFCS**  
- Run:
```bash
make
```
This will compile the source file `ILP_LFCS.cpp`.  
⚠️ Before running `make`, make sure to set the correct path to your CPLEX installation.

2. **Heuristics**  
- Copy the content of `Makefile_heuristics_literature.txt` into your working Makefile.  
- Then run:
```bash
make
```
This will compile the source file `Heuristics.cpp`.

3. **Executables**  
After these steps, you should have the following binaries:
- `ILP_LFCS`
- `Heuristics`

---

### Running the Heuristics

The `Heuristics` executable supports **RandomSample**, **Approx**, and **LS** approaches.

- **RandomSample**  

./Heuristics -a 0 -o out.out -i ../instances_and_generators/RANDOM/64_16_23.txt

-i: path to the instance file

-o: file where results will be stored

- **Approx**  
./Heuristics -a 1 -o out.out -i ../instances_and_generators/RANDOM/64_16_23.txt

- **Local serch**  
./Heuristics -a 2 -k 2 -o out.out -i ../instances_and_generators/RANDOM/64_16_23.txt


- **Pure ILP**  
./ILP_LFCS -a 0 -o out.out -i ../instances_and_generators/RANDOM/64_16_23.txt -t 600
where -t is the maximum allowed time. 

- **AdaptCMSA**  
./ILP_LFCS -a 2 -alpha_lb 0.6 -alpha_ub 1.0 -t_limit_ILP 10 -t_prop 0.5 -alpha_red 0.2 \
-o out.out -i ../instances_and_generators/RANDOM/64_16_23.txt -t 600
