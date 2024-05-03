OPTIMIZED_OPT=-O2 -fopenmp-simd -march=native -ffp-contract=fast -g -Wno-pass-failed  -fno-omit-frame-pointer -fPIC
# DEBUG_OPT=-O0 -fopenmp-simd -march=native -ffp-contract=fast -g -Wno-pass-failed -fno-omit-frame-pointer -fPIC -Wall
DEBUG_OPT=-std=c99 -fPIC -Wall -Wwrite-strings -Wno-unknown-pragmas -Wno-lto-type-mismatch -Wno-stringop-overflow -fstack-protector -g
OPT=${OPTIMIZED_OPT}
# PETSC_ARCH=arch-parallel-int64
PETSC_DIR=${HOME}/project/micromorph/petsc.worktrees/zach/snes-arc-length
CC=${PETSC_DIR}/${PETSC_ARCH}/bin/mpicc
CXX=${PETSC_DIR}/${PETSC_ARCH}/bin/mpicxx
COMPILER_VERSION=$(shell ${CC} --version | head -n 1)
ifneq (,$(findstring gcc,${COMPILER_VERSION}))
	ENZYME_LIB=
endif
ifneq (,$(filter-out undefined default,$(origin DEBUG)))
	OPT=${DEBUG_OPT}
	ENZYME_LIB=
endif
PETSC_OPTIONS=-malloc no
# LDFLAGS=
