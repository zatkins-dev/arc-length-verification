#pragma once

#include <petscdm.h>
#include <petscoptions.h>
#include <petscsnes.h>

#include "common.h"

typedef struct {
  PetscViewer       viewer;
  DM                dm;
  PetscInt          step;
  char             *filename;
  char             *extension;
  PetscBool         is_vtk;
  PetscBool         is_header_written;
  PetscViewerFormat format;
  PetscViewerType   type;
} *MonitorCtx;

PetscErrorCode MonitorCtxCreate(DM dm, const char option[], MonitorCtx *ctx);
PetscErrorCode MonitorCtxDestroy(void **ctx);

PetscErrorCode MonitorSolution(SNES snes, PetscInt its, PetscReal fnorm, void *ctx);
PetscErrorCode MonitorForce(SNES snes, PetscInt its, PetscReal fnorm, void *ctx);
PetscErrorCode MonitorForceDisplacement(SNES snes, PetscInt its, PetscReal fnorm, void *ctx);

PetscErrorCode TSMonitorSolutionC(TS ts, PetscInt step, PetscReal time, Vec u, void *ctx);
PetscErrorCode TSMonitorForce(TS ts, PetscInt step, PetscReal time, Vec u, void *ctx);
PetscErrorCode TSMonitorForceDisplacement(TS ts, PetscInt step, PetscReal time, Vec u, void *ctx);
