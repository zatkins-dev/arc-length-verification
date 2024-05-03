#pragma once

#include <petsc.h>

#define QP0 0.2113248654051871
#define QP1 0.7886751345948129
#define NQ 2
#define NB 2
#define NEB 8
#define NEQ 8
#define NPB 24

#define NVALS NEB *NEQ

typedef PetscScalar Field[3];
typedef PetscScalar CoordField[3];
typedef PetscScalar JacField[9];

typedef struct {
  PetscReal loading;
  PetscReal mu;
  PetscReal lambda;
  PetscReal rad;
  PetscReal height;
  PetscReal width;
  PetscReal arc;
  PetscReal ploading;
  PetscReal load_factor;
  PetscBool use_implicit;
  PetscInt  point_loc[3];
  DM        da;
} AppCtx;

typedef enum {
  CONTROL_LOAD,
  CONTROL_ARC_LENGTH,
} ControlType;
extern const char *const ControlTypes[];
extern const char *const ControlTypesCL[];

PetscErrorCode InitialGuess(DM, AppCtx *, Vec);
PetscErrorCode FormRHS(DM, AppCtx *, Vec);
PetscErrorCode FormCoordinates(DM, AppCtx *);
PetscErrorCode TangentLoad(SNES, Vec, Vec, void *);
PetscInt       OnPointForceBoundary(DM, PetscInt, PetscInt, PetscInt);
