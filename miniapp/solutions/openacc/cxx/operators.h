//******************************************
// operators.h
// based on min-app code written by Oliver Fuhrer, MeteoSwiss
// modified by Ben Cumming, CSCS
// *****************************************

// Description: Contains simple operators which can be used on 3d-meshes

#ifndef OPERATORS_H
#define OPERATORS_H

#include "data.h"

namespace operators
{
using data::Field;

// const qualifier issues a "cannot determine bounds for array" error in PGI
// void diffusion(const data::Field &U, data::Field &S)
void diffusion(Field& up, Field& sp);
}

#endif // OPERATORS_H
