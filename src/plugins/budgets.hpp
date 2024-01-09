#if !defined(nekrs_budgets_hpp_)
#define nekrs_budgets_hpp

#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"

namespace budgets
{
void buildKernel(occa::properties kernelInfo);
void run(dfloat time, bool comp_skew, bool comp_TKEbudget, bool comp_THFbudget);
void setup(nrs_t* nrs_, bool comp_skew, bool comp_TKEbudget, bool comp_THFbudget);
void outfld(bool comp_skew, bool comp_TKEbudget, bool comp_THFbudget);
void outfld(int outXYZ, int FP64, bool comp_skew, bool comp_TKEbudget, bool comp_THFbudget);
void reset();
void EX (dlong N, dfloat a, dfloat b, int nflds, occa::memory o_x, occa::memory o_EX);
void EXX(dlong N, dfloat a, dfloat b, int nflds, occa::memory o_x, occa::memory o_EXX);
void EXXX(dlong N, dfloat a, dfloat b, int nflds, occa::memory o_x, occa::memory o_EXXX);
void EXXXX(dlong N, dfloat a, dfloat b, int nflds, occa::memory o_x, occa::memory o_EXXXX);
void EXYsame(dlong N,
         dfloat a,
         dfloat b,
         int nflds,
         occa::memory o_x,
         occa::memory o_y,
         occa::memory o_EXYsame);
void EXYdiff(dlong N,
         dfloat a,
         dfloat b,
         int nflds_x,
         int nflds_y,
         occa::memory o_x,
         occa::memory o_y,
         occa::memory o_EXYdiff);
void EXXYsame(dlong N,
         dfloat a,
         dfloat b,
         int nflds,
         occa::memory o_x,
         occa::memory o_y,
         occa::memory o_EXXYsame);
void EXXYdiff(dlong N,
         dfloat a,
         dfloat b,
         int nflds_x,
         int nflds_y,
         occa::memory o_x,
         occa::memory o_y,
         occa::memory o_EXXYdiff);
void EXYZsame(dlong N,
              dfloat a,
              dfloat b,
              int nflds_xy,
              int nflds_z,
              occa::memory o_x,
              occa::memory o_y,
              occa::memory o_z,
              occa::memory o_EXYZsame);

}
#endif
