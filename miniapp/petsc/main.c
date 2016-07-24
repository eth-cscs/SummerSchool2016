/*******************************************
Implicit time stepping implementation of a 2D diffusion problem
Adapted to use PETSc by Patrick Sanan, from code by Ben Cumming and Gilles Fourestey

This code is heavily commented for the newcomer.

usage if you are using your own PETSc build:
$PETSC_DIR/bin/petscmpiexec -n <numprocs> ./main -nx <nx> -ny <ny> -nt <nt> -t <t> [-dump 1] [-assemble 1]
[Note that you should have PETSC_DIR and PETSC_ARCH defined in your environment in this case]

if using the cray-petsc module, run as any other mpi code, using aprun

*******************************************/

static char help[] = "Time-stepping for a nonlinear 2D Diffusion Equation, in parallel.\n\
Available options:\n\
-nx <nx> -ny <ny> : set number of grid points in the x and y directions.\n\
-nt <nt>          : set the number of time steps.\n\
-t <t>            : set the final time.\n\
-dump <1/0>       : dump output (uniprocessor only!).\n\
-assemble <1/0>   : use an assembled Jacobian (useful to experiment with preconditioners) \n\
\n";

#include <petscts.h>
#include <petscdm.h>
#include <petscdmda.h>

/* An Application Context
   This provides a way to pass information aboout the problem to solvers and
   other routines, without the overhead of C++
*/
#include "appctx.h"

/* Functions defined in other compilation units */
extern PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode RHSJacobianAssembled(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode InitialConditions(Vec,AppCtx*);
extern PetscErrorCode DumpSolution(DM,Vec,AppCtx*);
extern PetscErrorCode DumpSolutionMatlab(DM,Vec);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  /* PETSc uses ANSI C*, with its own structures to allow object-oriented
     design. The declarations below show many of the important object types:
     TS   : time stepper (e.g. Backwards Euler)
     SNES : nonlinear solver (e.g. Newton's method)
     KSP  : linear solver (e.g. Conjugate Gradients)
     Mat  : operator (e.g. diffusion stencil, or matrix multiply)
     Vec  : (distributed) vector (e.g. field on a domain)
     DM   : (distributed) domain manager (e.g. 2D grid)
     
     *It can of course be called from C++, and provides a Fortran 
     interface. */
  PetscErrorCode ierr; /* PETSc provides lightweight error checking and handling */
  AppCtx         ctx;
  DM             da;   
  TS             ts;
  SNES           snes;
  KSP            ksp;
  Mat            J;
  Vec            u;
  PetscInt       nt=100;
  PetscReal      t=0.01;
  PetscBool      assemble_jacobian=PETSC_FALSE, dump=PETSC_FALSE, dumpmatlab=PETSC_FALSE;

  /* Initialize PETSc, which initializes MPI if it has not already been,
     along with logs, a database of options, an error handler, etc. 
     Command line options, an optional options file, and a help string
     are provided
      */
  PetscInitialize(&argc,&argv,(char*)0,help);

  /* Default problem sizes. 
     Note that this is only done here to give an interface similar to the 
     previous examples. This information is stored by a DMDA object as well.
    */
  ctx.nx      = 128;
  ctx.ny      = 128;

  /* Process command line arguments.
     PETSc maintains an internal "options database" which stores
     string-indexed values like these. These can be used, as they are here,
     to define your own behaviour, and are also used by functions like
     KSPSetFromOptions(), SNESSetFromOptions(), etc. To access a huge number
     of parameters. Run your program with -help to see some of the available options.
     */
#ifdef USE_PETSC351_API
  ierr = PetscOptionsGetInt(NULL,"-nx",&ctx.nx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-ny",&ctx.ny,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-nt",&nt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,"-t",&t,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,"-assemble",&assemble_jacobian,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,"-dump",&dump,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,"-dumpmatlab",&dumpmatlab,NULL);CHKERRQ(ierr);
#else
  ierr = PetscOptionsGetInt(NULL,NULL,"-nx",&ctx.nx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-ny",&ctx.ny,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-nt",&nt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-t",&t,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-assemble",&assemble_jacobian,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-dump",&dump,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-dumpmatlab",&dumpmatlab,NULL);CHKERRQ(ierr);
#endif

  /* For convenience, compute and store 1/h^2 */
  ctx.dxinv2 = (ctx.nx-1.0)*(ctx.nx-1.0);

  /* Set up a Distributed Array Domain Manager (DMDA) 
    Note: in many PETSc examples you will see negative sizes in this function,
    meaning that they can be set from the command line with options like
    -da_grid_x <nx> . We do not use this functionality here to provide
    an interface closer to that of the base code.

    Note here that PETSc uses an MPI communicator called PETSC_COMM_WORLD.
    By default this is MPI_COMM_WORLD, but this can be changed, meaning
    that the library is safe to use within a larger MPI program.
   */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,ctx.nx,ctx.ny,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da);CHKERRQ(ierr);

  /* Specify uniform coordinates on the 2D domain. The number of points in the x direction, which is of unit width, determines the size of the square cells */
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,((PetscReal)ctx.ny)/ctx.nx,0,0);CHKERRQ(ierr);
 
  /* Store a reference to the DA in the context. DM and other PETSc objects are
     pointer types, so they may be used in this way */
  ctx.da=da;

  /* Use the Domain Manager to create an operator (Mat) and vector (Vec)
     with the appropriate parallel layout */
  ierr = DMCreateMatrix(da,&J);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u,"u");CHKERRQ(ierr);

  /* Define a "local" vector which includes halo points */
  ierr = DMCreateLocalVector(da,&ctx.u_local);CHKERRQ(ierr);

  /* Create a timestepper (TS) context */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);

  /* Set the RHS and Jacobian routines */
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,&ctx);CHKERRQ(ierr);
  if(assemble_jacobian){
   /* For instructive purposes, we assemble the Jacobian, but in practical use
      one would often like to simply define a function to apply it.
      This sort of "matrix free" operator is represented in PETSc by a 
      MatShell object */
    ierr = TSSetRHSJacobian(ts,J,J,RHSJacobianAssembled,&ctx);CHKERRQ(ierr);
  }else{
    /* Instruct the nonlinear solve to use a default finite-difference 
       approximation to estimate the Jacobian.
       Note that we do this by directly manipulating the options database,
       as this is a common procedure which is set with a special option.

       You can see the finite difference stepsize used by providing the
       -snes_mf_ksp_monitor option.
       Also note the -snes_mf_operator option if you would like
       to supply a preconditioner. */
#ifdef USE_PETSC351_API
     ierr = PetscOptionsSetValue("-snes_mf",NULL);CHKERRQ(ierr);
#else
     ierr = PetscOptionsSetValue(NULL,"-snes_mf",NULL);CHKERRQ(ierr);
#endif
  }

  /* Set the timestep and final time */
  ierr = TSSetTimeStep(ts,((PetscReal) t)/nt);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,nt,t);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

  /* Choose the method of time integration */
  ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr); /* Backwards Euler */

  /* Extract the nonliner solver and linear solver to set some parameters */
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESSetTolerances(snes,1e-6,1e-6,PETSC_DEFAULT,50,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1e-6,1e-6,PETSC_DEFAULT,200);CHKERRQ(ierr);

  /* Set from options. This allows this code to be more concise,
     as many other features can be used at runtime. For example, 
     try running with the -ts_monitor or -snes_monitor options. 
     You can see more options by running with -help */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* Set the Initial Conditions */
  ierr = InitialConditions(u,&ctx);CHKERRQ(ierr);
 
  /* Run */
  ierr = TSSolve(ts,u);CHKERRQ(ierr);

  /* Dump output */
  if(dump){
    ierr = DumpSolution(da,u,&ctx);CHKERRQ(ierr);
  }
  if(dumpmatlab){
    ierr = DumpSolutionMatlab(da,u);CHKERRQ(ierr);
  }

  /* Free allocated objects.
     C does not include destructors, so you must do this
     yourself to decrement reference counts and possibly
     free allocated resources.
    */
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx.u_local);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  
  PetscFinalize();

  return 0;
}
