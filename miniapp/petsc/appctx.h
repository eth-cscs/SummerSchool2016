#include <petscts.h>

/* A user-defined context to store problem data */
typedef struct {
  DM          da;             /* distributed array data structure */
  Vec         u_local;        /* local ghosted approximate solution vector */
  PetscInt    nx,ny;          /* Grid sizes */
  PetscScalar dxinv2;         /* holds 1/dx^2, which is (nx-1)^2 */
} AppCtx;
