#include <petscts.h>
#include <petscdm.h>
#include <petscdmda.h>
#include "appctx.h"

/* A function that assembles the Jacobian as a distributed sparse
   matrix */
#undef __FUNCT__
#define __FUNCT__ "AssembleJacobian"
static PetscErrorCode AssembleJacobian(Vec u,Mat J,void *ptr)
{
  AppCtx            *ctx = (AppCtx*)ptr;
  PetscErrorCode    ierr;
  PetscInt          i, j, M, N, xm, ym, xs, ys, num;
  PetscScalar       v[5];
  const PetscScalar **uarr;
  MatStencil        row, col[5];
  DM                da=ctx->da;

  PetscFunctionBeginUser;

#ifdef USE_PETSC351_API
  ierr = DMDAVecGetArray(da,u,&uarr);CHKERRQ(ierr);
#else
  ierr = DMDAVecGetArrayRead(da,u,&uarr);CHKERRQ(ierr);
#endif
  ierr = DMDAGetInfo(da,0,&M,&N,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);

  #if 0
  /* Print the local sizes */
  {
    PetscMPIInt rank,size;
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"proc id %d [%d total]. xs:%d ys:%d xm:%d ym:%d\n",rank,size,xs,ys,xm,ym);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
  }
  #endif

  /* Loop over indices assigned to this rank 
    Note that we have introduced inefficent branching into the assembly here
    for conciseness. An obvious optimization is to treat the boundary cases separately
    as you have done in the other exercises in this course */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;
      v[0] = -(4.0 * ctx->dxinv2) + (1.0 - 2.0*uarr[j][i]); 
      col[0].i = i; 
      col[0].j = j;
      num = 1;
      if (j!=0) {
        v[num] = ctx->dxinv2;                
        col[num].i = i;   
        col[num].j = j-1;
        ++num;
      }
      if (i!=0) {
        v[num] = ctx->dxinv2;                
        col[num].i = i-1; 
        col[num].j = j;
        ++num;
      }
      if (i!=M-1) {
        v[num] = ctx->dxinv2;                
        col[num].i = i+1; 
        col[num].j = j;
        ++num;
      }
      if (j!=N-1) {
        v[num] = ctx->dxinv2;                
        col[num].i = i;   
        col[num].j = j+1;
        ++num;
      }
      ierr = MatSetValuesStencil(J,1,&row,num,col,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
#ifdef USE_PETSC351_API
  ierr = DMDAVecRestoreArray(da,u,&uarr);CHKERRQ(ierr);
#else
  ierr = DMDAVecRestoreArrayRead(da,u,&uarr);CHKERRQ(ierr);
#endif
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RHSJacobianAssembled"
PetscErrorCode RHSJacobianAssembled(TS ts,PetscReal t,Vec u,Mat A,Mat P,void* ptr)
{
  PetscErrorCode    ierr;

  ierr = AssembleJacobian(u,A,ptr);CHKERRQ(ierr);
  if(A != P){
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Separate PC mat not supported");
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RHSFunction"
PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec u,Vec f,void* ptr)
{
  AppCtx            *ctx = (AppCtx*)ptr;
  PetscErrorCode    ierr;
  PetscInt          i,j,M,N,xm,ym,xs,ys;
  const PetscScalar **uarr;
  PetscScalar       **farr;
  DM                da=ctx->da;
  Vec               u_local=ctx->u_local;

  PetscFunctionBeginUser;

  /* Scatter global-->local to have access to the required ghost values */
  ierr=DMGlobalToLocalBegin(da,u,INSERT_VALUES,u_local);CHKERRQ(ierr);
  ierr=DMGlobalToLocalEnd  (da,u,INSERT_VALUES,u_local);CHKERRQ(ierr);
  
  /* Obtain access to the raw arrays, indexed with *global coordinates*
     Note that we access the "local" version of u which we just populated,
     meaning that we can access the required "ghost" or "halo" data
     which was just obtained from neighboring processes via MPI calls.
    */
#ifdef USE_PETSC351_API
  ierr = DMDAVecGetArray(da,u_local,&uarr);CHKERRQ(ierr);
#else
  ierr = DMDAVecGetArrayRead(da,u_local,&uarr);CHKERRQ(ierr);
#endif
  ierr  = DMDAVecGetArray(da,f,&farr);CHKERRQ(ierr);
  ierr  = DMDAGetInfo(da,0,&M,&N,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr  = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);

  /* Loop over indices assigned to this rank 
    Note that we have introduced inefficent branching into the assembly here
    for conciseness. An obvious optimization is to treat the boundary cases separately
    as you have done in the other exercises in this course */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      PetscScalar val = (-(4.0 * ctx->dxinv2) + (1.0 - uarr[j][i]))*uarr[j][i]; /* nonlinearity */
      if (j!=0) {
         val += ctx->dxinv2 * uarr[j-1][i];
      }
      if (i!=0) {
        val += ctx->dxinv2 *uarr[j][i-1];
      }
      if (i!=M-1) {
        val += ctx->dxinv2 *uarr[j][i+1];
      }
      if (j!=N-1) {
        val += ctx->dxinv2 *uarr[j+1][i];
      }
      farr[j][i] = val;
    }
  }
#ifdef USE_PETSC351_API
  ierr = DMDAVecRestoreArray(da,u_local,&uarr);CHKERRQ(ierr);
#else
  ierr = DMDAVecRestoreArrayRead(da,u_local,&uarr);CHKERRQ(ierr);
#endif
  ierr  = DMDAVecRestoreArray(da,f,&farr);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "InitialConditions"
PetscErrorCode InitialConditions(Vec u0,void *ptr)
{
  PetscErrorCode ierr;
  AppCtx         *ctx = (AppCtx*) ptr;
  PetscScalar    **u0arr;
  DM             da=ctx->da, cda;
  DMDALocalInfo  info;
  Vec            coordinates;
  DMDACoor2d     **coords;
  PetscInt       i,j;

  PetscFunctionBeginUser;

  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

  /* Get Coordinate DA, which can provide the coordinates for the grid points */
  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&coordinates);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,coordinates,&coords);CHKERRQ(ierr);
 
  ierr = DMDAVecGetArray(da,u0,&u0arr);CHKERRQ(ierr); 
    const PetscReal xc = 1.0 / 4.0;
    const PetscReal yc = ((PetscReal) ctx->ny - 1) / (4.0 * (ctx->nx - 1)); 
    const PetscReal radius = PetscMin(xc, yc) / 2.0;
    for (j=info.ys; j<info.ys+info.ym; ++j) {
      for (i=info.xs; i<info.xs+info.xm; ++i) {
        const PetscReal x = coords[j][i].x; 
        const PetscReal y = coords[j][i].y; 
        if ((x - xc) * (x - xc) + (y - yc) * (y - yc) < radius * radius){
          u0arr[j][i]=0.1;
        }else{
          u0arr[j][i]=0.0;
        }
      }
    }

    ierr = DMDAVecRestoreArray(da,u0,&u0arr);CHKERRQ(ierr); 
    ierr = DMDAVecRestoreArray(cda,coordinates,&coords);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}
