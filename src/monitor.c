#include "monitor.h"

PetscErrorCode MonitorCtxCreate(DM dm, const char option[], MonitorCtx *ctx) {
  PetscViewer       viewer;
  const char       *file_name;
  PetscViewerFormat format;
  PetscBool         set;

  PetscFunctionBeginUser;

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Monitor Options", NULL);
  PetscCall(PetscOptionsViewer(option, "Monitor file", NULL, &viewer, &format, &set));
  PetscOptionsEnd();
  if (!set) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "No monitor file specified\n"));
    *ctx = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscNew(ctx));

  (*ctx)->viewer = viewer;
  PetscCall(PetscViewerGetType(viewer, &(*ctx)->type));
  (*ctx)->format = format;

  // Set header default
  (*ctx)->is_header_written = PETSC_FALSE;

  PetscCall(PetscViewerFileGetName(viewer, &file_name));
  if (file_name) {
    char *file_extension;

    PetscCall(PetscStrrchr(file_name, '.', &file_extension));
    {
      PetscBool is_vtu = PETSC_FALSE, is_vtk = PETSC_FALSE, is_vts = PETSC_FALSE;

      PetscCall(PetscStrncmp(file_extension, "vtu", 4, &is_vtu));
      PetscCall(PetscStrncmp(file_extension, "vtk", 4, &is_vtk));
      PetscCall(PetscStrncmp(file_extension, "vts", 4, &is_vts));
      (*ctx)->is_vtk = is_vtu || is_vtk || is_vts;
    }
    if ((*ctx)->is_vtk) {
      PetscSizeT file_name_len      = file_extension - file_name;
      PetscSizeT file_extension_len = 0;

      PetscCall(PetscStrlen(file_extension, &file_extension_len));
      file_extension_len += 1;
      PetscCall(PetscCalloc1(file_name_len + 1, &(*ctx)->filename));
      PetscCall(PetscSNPrintf((*ctx)->filename, file_name_len, "%s", file_name));
      PetscCall(PetscCalloc1(file_extension_len + 1, &(*ctx)->extension));
      PetscCall(PetscSNPrintf((*ctx)->extension, file_extension_len, "%s", file_extension));
    }
    {
      PetscBool is_csv = PETSC_FALSE;

      PetscCall(PetscStrncmp(file_extension, "csv", 4, &is_csv));
      if (is_csv) (*ctx)->format = PETSC_VIEWER_ASCII_CSV;
    }
    if (!(*ctx)->filename) {
      PetscCall(PetscCalloc1(strlen(file_name) + 1, &(*ctx)->filename));
      PetscCall(PetscStrcpy((*ctx)->filename, file_name));
      if (file_extension) {
        PetscCall(PetscCalloc1(strlen(file_extension) + 1, &(*ctx)->extension));
        PetscCall(PetscStrcpy((*ctx)->extension, file_extension));
      } else (*ctx)->extension = NULL;
    }
  } else {
    (*ctx)->filename  = NULL;
    (*ctx)->extension = NULL;
  }

  PetscCall(PetscObjectReference((PetscObject)dm));
  (*ctx)->dm = dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MonitorCtxDestroy(void **ctx) {
  MonitorCtx *monitor_ctx;

  PetscFunctionBeginUser;
  if (!ctx) PetscFunctionReturn(PETSC_SUCCESS);
  monitor_ctx = (MonitorCtx *)(ctx);
  {
    PetscBool is_stdout = PETSC_FALSE;

    if ((*monitor_ctx)->filename) PetscCall(PetscStrncmp((*monitor_ctx)->filename, "stdout", 7, &is_stdout));
    if (!is_stdout) PetscCall(PetscViewerDestroy(&(*monitor_ctx)->viewer));
  }
  PetscCall(DMDestroy(&(*monitor_ctx)->dm));
  PetscCall(PetscFree((*monitor_ctx)->filename));
  PetscCall(PetscFree((*monitor_ctx)->extension));
  PetscCall(PetscFree(*ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeForceLoadControl(TS ts, Vec U, Vec F) {
  AppCtx   *user;
  DM        dm;
  PetscReal t;
  Vec       loadF;
  SNES      snes;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(TSGetTime(ts, &t));
  PetscCall(DMGetApplicationContext(dm, (void **)&user));
  PetscCall(FormRHS(dm, user, F));
  PetscCall(VecScale(F, t));

  PetscCall(DMGetGlobalVector(dm, &loadF));
  PetscCall(VecZeroEntries(loadF));
  PetscCall(TSGetSNES(ts, &snes));
  PetscCall(TangentLoad(snes, U, loadF, user));
  // Fix the "load factor"
  user->load_factor = t;
  PetscCall(VecAXPY(F, user->load_factor, loadF));
  PetscCall(DMRestoreGlobalVector(dm, &loadF));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MonitorVecCommon(DM dm, PetscInt step, PetscReal t, Vec X, MonitorCtx monitor_ctx) {
  PetscInt  old_step;
  PetscReal old;

  PetscFunctionBeginUser;
  PetscCall(DMGetOutputSequenceNumber(dm, &old_step, &old));
  PetscCall(DMSetOutputSequenceNumber(dm, step, t));

  // Update filename
  if (monitor_ctx->is_vtk) {
    char output_filename[PETSC_MAX_PATH_LEN];

    PetscCall(PetscSNPrintf(output_filename, sizeof(output_filename), "%s_%" PetscInt_FMT ".%s", monitor_ctx->filename, monitor_ctx->step,
                            monitor_ctx->extension));
    PetscCall(PetscViewerFileSetName(monitor_ctx->viewer, output_filename));
  }

  PetscCall(PetscViewerPushFormat(monitor_ctx->viewer, monitor_ctx->format));
  PetscCall(VecView(X, monitor_ctx->viewer));
  PetscCall(PetscViewerPopFormat(monitor_ctx->viewer));

  PetscCall(DMSetOutputSequenceNumber(dm, old_step, old));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MonitorSolution(SNES snes, PetscInt its, PetscReal fnorm, void *ctx) {
  MonitorCtx          monitor_ctx = (MonitorCtx)ctx;
  DM                  dm          = monitor_ctx->dm;
  Vec                 u;
  PetscReal           lambda;
  SNESConvergedReason reason;

  PetscFunctionBeginUser;
  PetscCall(SNESGetConvergedReason(snes, &reason));
  if (reason <= 0) PetscFunctionReturn(PETSC_SUCCESS);

  // Get solution
  PetscCall(SNESGetSolution(snes, &u));
  PetscCall(SNESNewtonALGetLoadParameter(snes, &lambda));
  PetscCall(PetscObjectSetName((PetscObject)u, "Solution"));

  // Monitor vector
  PetscCall(MonitorVecCommon(dm, monitor_ctx->step++, lambda, u, monitor_ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MonitorForce(SNES snes, PetscInt its, PetscReal fnorm, void *ctx) {
  MonitorCtx          monitor_ctx = (MonitorCtx)ctx;
  DM                  dm          = monitor_ctx->dm;
  Vec                 u, Q;
  PetscReal           lambda;
  SNESConvergedReason reason;

  PetscFunctionBeginUser;
  PetscCall(SNESGetConvergedReason(snes, &reason));
  if (reason <= 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(SNESGetSolution(snes, &u));

  // Get force
  PetscCall(DMGetNamedGlobalVector(dm, "force", &Q));
  PetscCall(SNESNewtonALComputeFunction(snes, u, Q));
  PetscCall(SNESNewtonALGetLoadParameter(snes, &lambda));
  PetscCall(VecScale(Q, lambda));
  PetscCall(PetscObjectSetName((PetscObject)Q, "Force"));

  // Monitor vector
  PetscCall(MonitorVecCommon(dm, monitor_ctx->step++, lambda, Q, monitor_ctx));

  // Cleanup
  PetscCall(DMRestoreNamedGlobalVector(dm, "force", &Q));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MonitorForceDisplacement(SNES snes, PetscInt its, PetscReal fnorm, void *ctx) {
  MonitorCtx          monitor_ctx = (MonitorCtx)ctx;
  DM                  da          = monitor_ctx->dm;
  PetscInt            i, j, k, xs, ys, zs, xm, ym, zm;
  PetscInt            mx, my, mz;
  Field            ***u, ***f;
  PetscReal           lambda, force;
  Field               displacement;
  Vec                 U, F;
  SNESConvergedReason reason;

  PetscFunctionBegin;
  if (!(monitor_ctx->is_header_written)) {
    if (monitor_ctx->format == PETSC_VIEWER_ASCII_CSV) {
      PetscCall(PetscViewerASCIIPrintf(monitor_ctx->viewer, "step,LPF,force,u_x,u_y\n"));
      PetscCall(PetscViewerASCIIPrintf(monitor_ctx->viewer, "%" PetscInt_FMT ",%g,%0.12e,%0.12e,%0.12e\n", monitor_ctx->step, 0.0, 0.0, 0.0, 0.0));
    } else {
      PetscCall(PetscViewerASCIIPrintf(monitor_ctx->viewer, "%" PetscInt_FMT " LPF %g Force %0.12e Displacement %0.12e\n", monitor_ctx->step, 0.0,
                                       0.0, 0.0));
    }
    monitor_ctx->is_header_written = PETSC_TRUE;
  }
  PetscCall(SNESGetConvergedReason(snes, &reason));
  if (reason <= 0) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(SNESGetSolution(snes, &U));
  PetscCall(DMGetNamedGlobalVector(da, "force", &F));
  PetscCall(SNESNewtonALComputeFunction(snes, U, F));
  PetscCall(SNESNewtonALGetLoadParameter(snes, &lambda));
  PetscCall(VecScale(F, lambda));
  monitor_ctx->step++;

  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));
  PetscCall(DMDAGetInfo(da, 0, &mx, &my, &mz, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCall(DMDAVecGetArrayRead(da, U, &u));
  PetscCall(DMDAVecGetArrayRead(da, F, &f));

  for (k = zs; k < zs + zm; k++) {
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        if (OnPointForceBoundary(da, i, j, k) && k == (mz - 1) / 2 && j == (my - 1)) {
          force = f[k][j][i][1];
          for (PetscInt l = 0; l < 2; l++) displacement[l] = u[k][j][i][l];
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(da, U, &u));
  PetscCall(DMDAVecRestoreArrayRead(da, F, &f));
  PetscCall(DMRestoreNamedGlobalVector(da, "force", &F));

  if (monitor_ctx->format == PETSC_VIEWER_ASCII_CSV) {
    PetscCall(PetscViewerASCIIPrintf(monitor_ctx->viewer, "%" PetscInt_FMT ",%g,%0.12e,%0.12e,%0.12e\n", monitor_ctx->step, lambda, force,
                                     displacement[0], -displacement[1]));
  } else {
    PetscCall(PetscViewerASCIIPrintf(monitor_ctx->viewer, "%" PetscInt_FMT " t=%g Force %0.12e Displacement %0.12e,%0.12e\n", monitor_ctx->step,
                                     lambda, force, displacement[0], -displacement[1]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSMonitorSolutionC(TS ts, PetscInt step, PetscReal time, Vec U, void *ctx) {
  MonitorCtx monitor_ctx = (MonitorCtx)ctx;
  DM         da          = monitor_ctx->dm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectSetName((PetscObject)U, "Solution"));
  PetscCall(MonitorVecCommon(da, monitor_ctx->step++, time, U, monitor_ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSMonitorForce(TS ts, PetscInt step, PetscReal time, Vec U, void *ctx) {
  MonitorCtx monitor_ctx = (MonitorCtx)ctx;
  Vec        F;

  PetscFunctionBeginUser;
  PetscCall(DMGetNamedGlobalVector(monitor_ctx->dm, "force", &F));
  PetscCall(ComputeForceLoadControl(ts, U, F));
  PetscCall(MonitorVecCommon(monitor_ctx->dm, monitor_ctx->step++, time, F, monitor_ctx));
  PetscCall(DMRestoreNamedGlobalVector(monitor_ctx->dm, "force", &F));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSMonitorForceDisplacement(TS ts, PetscInt step, PetscReal time, Vec U, void *ctx) {
  MonitorCtx monitor_ctx = (MonitorCtx)ctx;
  DM         da          = monitor_ctx->dm;
  PetscInt   i, j, k, xs, ys, zs, xm, ym, zm;
  PetscInt   mx, my, mz;
  Field   ***u, ***f;
  PetscReal  force;
  Field      displacement;
  Vec        F;

  PetscFunctionBeginUser;
  PetscCall(DMGetNamedGlobalVector(da, "force", &F));
  PetscCall(ComputeForceLoadControl(ts, U, F));
  monitor_ctx->step++;

  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));
  PetscCall(DMDAGetInfo(da, 0, &mx, &my, &mz, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCall(DMDAVecGetArrayRead(da, U, &u));
  PetscCall(DMDAVecGetArrayRead(da, F, &f));

  for (k = zs; k < zs + zm; k++) {
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        if (i == (mx - 1) / 2 && j == (my - 1) && k == (mz - 1) / 2) {
          force = f[k][j][i][1];
          for (PetscInt l = 0; l < 2; l++) displacement[l] = u[k][j][i][l];
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(da, U, &u));
  PetscCall(DMDAVecRestoreArrayRead(da, F, &f));
  PetscCall(DMRestoreNamedGlobalVector(da, "force", &F));

  if (monitor_ctx->format == PETSC_VIEWER_ASCII_CSV) {
    if (!(monitor_ctx->is_header_written)) {
      PetscCall(PetscViewerASCIIPrintf(monitor_ctx->viewer, "step,time,force,u_x,u_y\n"));
      monitor_ctx->is_header_written = PETSC_TRUE;
    }
    PetscCall(PetscViewerASCIIPrintf(monitor_ctx->viewer, "%" PetscInt_FMT ",%g,%0.12e,%0.12e,%0.12e\n", monitor_ctx->step, time, -force,
                                     displacement[0], -displacement[1]));
  } else {
    PetscCall(PetscViewerASCIIPrintf(monitor_ctx->viewer, "%" PetscInt_FMT " t=%g Force %0.12e Displacement %0.12e,%0.12e\n", monitor_ctx->step, time,
                                     -force, displacement[0], -displacement[1]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
