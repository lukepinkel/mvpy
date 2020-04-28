#ifndef _CS_H
#define _CS_H
#include <stdlib.h>
// #include <stdint.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stddef.h>
#ifdef MATLAB_MEX_FILE
#include "mex.h"
#endif
#define CS_VER 3                    /* CSparse Version */
#define CS_SUBVER 1
#define CS_SUBSUB 2
#define CS_DATE "April 16, 2013"    /* CSparse release date */
#define CS_COPYRIGHT "Copyright (c) Timothy A. Davis, 2006-2013"

#ifdef MATLAB_MEX_FILE
#undef csi
#define csi mwSignedIndex
#endif
// #ifndef csi
// #define csi ptrdiff_t
// #endif
#include <stdint.h> // which will use the C99 header

// FORCE use of 32 bit int offsets, because thats what scipy uses so we can
// avoid a copy.
#define csi int32_t

/* --- primary CSparse routines and data structures ------------------------- */
typedef struct cs_sparse    /* matrix in compressed-column or triplet form */
{
    csi nzmax ;     /* maximum number of entries */
    csi m ;         /* number of rows */
    csi n ;         /* number of columns */
    csi *p ;        /* column pointers (size n+1) or col indices (size nzmax) */
    csi *i ;        /* row indices, size nzmax */
    double *x ;     /* numerical values, size nzmax */
    csi nz ;        /* # of entries in triplet matrix, -1 for compressed-col */
} cs ;

csi cs_gaxpy (const cs *A, const double *x, double *y) ;

#define CS_CSC(A) (A && (A->nz == -1))
#endif
