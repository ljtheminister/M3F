/* 
  MEXCOMMON File for shared constants and memory

  ------------------------------------------------------------------------     

  Last revision: 6-July-2010

  Authors: Lester Mackey and David Weiss
  License: MIT License

  Copyright (c) 2010 Lester Mackey & David Weiss

  Permission is hereby granted, free of charge, to any person obtaining
  a copy of this software and associated documentation files (the
  "Software"), to deal in the Software without restriction, including
  without limitation the rights to use, copy, modify, merge, publish,
  distribute, sublicense, and/or sell copies of the Software, and to
  permit persons to whom the Software is furnished to do so, subject to
  the following conditions:
  
  The above copyright notice and this permission notice shall be
  included in all copies or substantial portions of the Software.
  
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

  ------------------------------------------------------------------------
*/
#include "mexCommon.h"
#include "math.h"

// Return array of RNGs for multithreaded random number generation
gsl_rng** getRngArray(){
   static gsl_rng* rngs[MAX_NUM_THREADS];
   static int firstCall = 1;

   if(firstCall){
      // Initialize rngs
      gsl_rng_env_setup();
      for(unsigned long r = 0; r < MAX_NUM_THREADS; r++){
	 rngs[r] = gsl_rng_alloc(gsl_rng_taus);
	 gsl_rng_set(rngs[r], 2*r+1);
      }
      firstCall = 0;
   }

   return rngs;
}

// Extract user examples and lengths from this jagged cell array
// and store in preallocated arrays
// Function written from perspective of users
// Switch roles of user-item inputs for items
void unpackJagged(const mxArray* exampsByUser, uint32_t*** userExamps, 
		  mwSize** userLens, int numUsers) {
   *userExamps = mxMalloc(numUsers*sizeof(**userExamps));
   *userLens = mxMalloc(numUsers*sizeof(**userLens));
   for(int u = 0; u < numUsers; u++){
      mxArray* exampsArray = mxGetCell(exampsByUser, u);
      (*userExamps)[u] = (uint32_t*) mxGetData(exampsArray);
      (*userLens)[u] = mxGetN(exampsArray);      
   }
}

#ifndef NDEBUG
// Fill each entry of double array with val
void fillArrayD(double* array, const int len, const double val){
   for(int ii = 0; ii < len; ii++)
      array[ii] = val;
}

// Fill each entry of int array with val
void fillArrayI(int* array, const int len, const int val){
   for(int ii = 0; ii < len; ii++)
      array[ii] = val;
}

// Replace each entry of array with its logarithm
void logArrayD(double* array, const int len){
   for(int ii = 0; ii < len; ii++)
      array[ii] = log(array[ii]);
}
#endif


