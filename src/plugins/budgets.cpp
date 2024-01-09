/*
     compute running averages E(X), E(X*X)
     and E(X*Y) for velocity only

     statistics can be obtained by:

     avg(X)   := E(X)
     var(X)   := E(X*X) - E(X)*E(X)
     cov(X,Y) := E(X*Y) - E(X)*E(Y)

     Note: The E-operator is linear, in the sense that the expected
           value is given by E(X) = 1/N * avg[ E(X)_i ], where E(X)_i
           is the expected value of the sub-ensemble i (i=1...N).
 */


//////////NEW//////////

/*
     Initial functions to perform 3rd- and 4th-order statistics,
     TKE budgets, and THF budgets have been added. Testing needs to be performed.
     
     Notes:
       - No mapping of pressure mesh so PN-PN only
       - 3D only (maybe set up error message and abort if 2D)
       - Only set up for 2 scalars right now
*/

//////////NEW//////////

#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "budgets.hpp"
#include "platform.hpp"
#include "linAlg.hpp"

// private members
namespace
{
static ogs_t* ogs;
static nrs_t* nrs;

//////////////OCCA FIELDS///////////////

// standard
static occa::memory o_U_avg, o_U2_avg;
static occa::memory o_u_cov, o_v_cov, o_w_cov;
static occa::memory o_uu_avg, o_vv_avg, o_ww_avg;
static occa::memory o_vx, o_vy, o_vz;
static occa::memory o_uS_avg, o_vS_avg, o_wS_avg;
static occa::memory o_P_avg, o_P2_avg;
static occa::memory o_S_avg, o_S2_avg;

// derivatives
// ARK replace with # NScalar fields instead of hard code 
static occa::memory o_dvx, o_dvy, o_dvz;
static occa::memory o_s0, o_s1;
static occa::memory o_ds0, o_ds1;

// if higher-order
static occa::memory o_U3_avg, o_U4_avg;
static occa::memory o_P3_avg, o_P4_avg;
static occa::memory o_S3_avg, o_S4_avg;

// if TKE budget
static occa::memory o_dxu2_avg, o_dxv2_avg, o_dxw2_avg;
static occa::memory o_PdxU_avg, o_PU_avg;
static occa::memory o_u2U_avg, o_v2U_avg, o_w2U_avg;

// if THF budget
static occa::memory o_PS_avg;
static occa::memory o_dxuS_avg, o_dxvS_avg, o_dxwS_avg;
static occa::memory o_udxS_avg, o_vdxS_avg, o_wdxS_avg;
static occa::memory o_PdxS_avg;
static occa::memory o_Sdxu_avg, o_Sdxv_avg, o_Sdxw_avg;
static occa::memory o_UUS_avg, o_uvS_avg;
static occa::memory o_SSU_avg;

//////////OCCA KERNELS//////////

// standard
static occa::kernel EXKernel;
static occa::kernel EXXKernel;
static occa::kernel EXYsameKernel;
static occa::kernel EXYdiffKernel;

// if higher-order
static occa::kernel EXXXKernel;
static occa::kernel EXXXXKernel;

// if TKE/THF budget
static occa::kernel EXXYsameKernel;
static occa::kernel EXXYdiffKernel;
static occa::kernel EXYZsameKernel;

//////////OTHER//////////

static bool buildKernelCalled = 0;
static bool setupCalled = 0;

static int counter = 0;

static dfloat atime;
static dfloat timel;

static int outfldCounter = 0;
}

void budgets::buildKernel(occa::properties kernelInfo)
{
  const std::string path = getenv("NEKRS_KERNEL_DIR") + std::string("/plugins/");
  std::string kernelName, fileName;
  const std::string extension = ".okl";
  {
    kernelName = "EX";
    fileName = path + kernelName + extension;
    EXKernel = platform->device.buildKernel(fileName, kernelInfo, true);

//ARK Kernels
    kernelName = "EXX";
    fileName = path + kernelName + extension;
    EXXKernel = platform->device.buildKernel(fileName, kernelInfo, true);

    kernelName = "EXYsame";
    fileName = path + kernelName + extension;
    EXYsameKernel = platform->device.buildKernel(fileName, kernelInfo, true);

    kernelName = "EXYdiff";
    fileName = path + kernelName + extension;
    EXYdiffKernel = platform->device.buildKernel(fileName, kernelInfo, true);

    kernelName = "EXXX";
    fileName = path + kernelName + extension;
    EXXXKernel = platform->device.buildKernel(fileName, kernelInfo, true);

    kernelName = "EXXXX";
    fileName = path + kernelName + extension;
    EXXXXKernel = platform->device.buildKernel(fileName, kernelInfo, true);

    kernelName = "EXXYsame";
    fileName = path + kernelName + extension;
    EXXYsameKernel = platform->device.buildKernel(fileName, kernelInfo, true);

    kernelName = "EXXYdiff";
    fileName = path + kernelName + extension;
    EXXYdiffKernel = platform->device.buildKernel(fileName, kernelInfo, true);

    kernelName = "EXYZsame";
    fileName = path + kernelName + extension;
    EXYZsameKernel = platform->device.buildKernel(fileName, kernelInfo, true);
  }
  buildKernelCalled = 1;

}

void budgets::reset()
{
  counter = 0;
  atime   = 0;
}

// QUESTION: since these functions are defined with kernels based only on nrs->fieldOffset,
//           does that mean they are only good when Vmesh=Smesh?
void budgets::EX (dlong N, dfloat a, dfloat b, int nflds, occa::memory o_x, occa::memory o_EX)
{
  EXKernel(N, nrs->fieldOffset, nflds, a, b, o_x, o_EX);
}

void budgets::EXX(dlong N, dfloat a, dfloat b, int nflds, occa::memory o_x, occa::memory o_EXX)
{
  EXXKernel(N, nrs->fieldOffset, nflds, a, b, o_x, o_EXX);
}

void budgets::EXXX(dlong N, dfloat a, dfloat b, int nflds, occa::memory o_x, occa::memory o_EXXX)
{
  EXXXKernel(N, nrs->fieldOffset, nflds, a, b, o_x, o_EXXX);
}

void budgets::EXXXX(dlong N, dfloat a, dfloat b, int nflds, occa::memory o_x, occa::memory o_EXXXX)
{
  EXXXXKernel(N, nrs->fieldOffset, nflds, a, b, o_x, o_EXXXX);
}

//ARK
//amended to have different nflds for X and Y
void budgets::EXYsame(dlong N,
              dfloat a,
              dfloat b,
              int nflds,
              occa::memory o_x,
              occa::memory o_y,
              occa::memory o_EXYsame)
{
  EXYsameKernel(N, nrs->fieldOffset, nflds, a, b, o_x, o_y, o_EXYsame);
}

void budgets::EXYdiff(dlong N,
              dfloat a,
              dfloat b,
              int nflds_x,
              int nflds_y,
              occa::memory o_x,
              occa::memory o_y,
              occa::memory o_EXYdiff)
{
  EXYdiffKernel(N, nrs->fieldOffset, nflds_x, nflds_y, a, b, o_x, o_y, o_EXYdiff);
}

void budgets::EXXYsame(dlong N,
              dfloat a,
              dfloat b,
              int nflds,
              occa::memory o_x,
              occa::memory o_y,
              occa::memory o_EXXYsame)
{
  EXXYsameKernel(N, nrs->fieldOffset, nflds, a, b, o_x, o_y, o_EXXYsame);
}

void budgets::EXXYdiff(dlong N,
              dfloat a,
              dfloat b,
              int nflds_x,
              int nflds_y,
              occa::memory o_x,
              occa::memory o_y,
              occa::memory o_EXXYdiff)
{
  EXXYdiffKernel(N, nrs->fieldOffset, nflds_x, nflds_y, a, b, o_x, o_y, o_EXXYdiff);
}

void budgets::EXYZsame(dlong N,
              dfloat a,
              dfloat b,
              int nflds_xy,
              int nflds_z,
              occa::memory o_x,
              occa::memory o_y,
              occa::memory o_z,
              occa::memory o_EXYZsame)
{
  EXYZsameKernel(N, nrs->fieldOffset, nflds_xy, nflds_z, a, b, o_x, o_y, o_z, o_EXYZsame);
}

void budgets::run(dfloat time, bool comp_skew, bool comp_TKEbudget, bool comp_THFbudget)
{

  if(!nrs->timeStepConverged) return;

  nrsCheck(!setupCalled || !buildKernelCalled, MPI_COMM_SELF, EXIT_FAILURE,
         "%s\n", "called prior to tavg::setup()!");

  if(!counter) {
    atime = 0;
    timel = time;
  }
  counter++;

  const dfloat dtime = time - timel;
  atime += dtime;

  if(atime == 0 || dtime == 0) return;

  const dfloat b = dtime / atime;
  const dfloat a = 1 - b;

  mesh_t* mesh = nrs->meshV;
  const dlong N = mesh->Nelements * mesh->Np;

//////////BASELINE//////////

  // velocity
  EX (N, a, b, nrs->NVfields, nrs->o_U, o_U_avg);
  EXX(N, a, b, nrs->NVfields, nrs->o_U, o_U2_avg);

  const dlong offsetByte = nrs->fieldOffset * sizeof(dfloat);
  o_vx = nrs->o_U + 0 * offsetByte;
  o_vy = nrs->o_U + 1 * offsetByte;
  o_vz = nrs->o_U + 2 * offsetByte;
  o_uu_avg = o_U2_avg + 0 * offsetByte;
  o_vv_avg = o_U2_avg + 1 * offsetByte;
  o_ww_avg = o_U2_avg + 2 * offsetByte;

//  occa::memory &o_dvx = platform->o_mempool.slice0;
//  occa::memory &o_dvy = platform->o_mempool.slice3;
//  occa::memory &o_dvz = platform->o_mempool.slice6;
//  occa::memory &o_ds0 = platform->o_mempool.slice9;
//  occa::memory &o_ds1 = platform->o_mempool.slice12;

  o_u_cov.copyFrom(o_uu_avg, mesh->Nlocal*sizeof(dfloat), 0*offsetByte);
  EXYsame(N, a, b, 1, o_vx, o_vy, o_u_cov + 1 * offsetByte);
  EXYsame(N, a, b, 1, o_vx, o_vz, o_u_cov + 2 * offsetByte);
  o_v_cov.copyFrom(o_u_cov + 1 * offsetByte, mesh->Nlocal*sizeof(dfloat), 0*offsetByte);
  o_v_cov.copyFrom(o_vv_avg, mesh->Nlocal*sizeof(dfloat), 1*offsetByte);
  EXYsame(N, a, b, 1, o_vy, o_vz, o_v_cov + 2 * offsetByte);
  o_w_cov.copyFrom(o_u_cov + 2 * offsetByte, mesh->Nlocal*sizeof(dfloat), 0*offsetByte);
  o_w_cov.copyFrom(o_v_cov + 2 * offsetByte, mesh->Nlocal*sizeof(dfloat), 1*offsetByte);
  o_w_cov.copyFrom(o_ww_avg, mesh->Nlocal*sizeof(dfloat), 2*offsetByte);

  // pressure
  EX (N, a, b, 1, nrs->o_P, o_P_avg);
  EXX(N, a, b, 1, nrs->o_P, o_P2_avg);

  // scalars
  if(nrs->Nscalar) {

    cds_t* cds = nrs->cds;
    const dlong N = cds->mesh[0]->Nelements * cds->mesh[0]->Np;
    EX (N, a, b, cds->NSfields, cds->o_S, o_S_avg);
    EXX(N, a, b, cds->NSfields, cds->o_S, o_S2_avg);
    EXYdiff(N, a, b, 1, cds->NSfields, o_vx, cds->o_S, o_uS_avg);
    EXYdiff(N, a, b, 1, cds->NSfields, o_vy, cds->o_S, o_vS_avg);
    EXYdiff(N, a, b, 1, cds->NSfields, o_vz, cds->o_S, o_wS_avg);
    EXYdiff(N, a, b, 1, cds->NSfields, nrs->o_P, cds->o_S, o_PS_avg);
  }

//////////HIGHER-ORDER//////////

  if(comp_skew) {
    
    const dlong N = mesh->Nelements * mesh->Np;

      // velocity
    EXXX (N, a, b, nrs->NVfields, nrs->o_U, o_U3_avg);
    EXXXX(N, a, b, nrs->NVfields, nrs->o_U, o_U4_avg);

    // pressure
    EXXX (N, a, b, 1, nrs->o_P, o_P3_avg);
    EXXXX(N, a, b, 1, nrs->o_P, o_P4_avg);

    // scalars
    if(nrs->Nscalar) {
      cds_t* cds = nrs->cds;
      const dlong N = cds->mesh[0]->Nelements * cds->mesh[0]->Np;
      EXXX (N, a, b, cds->NSfields, cds->o_S, o_S3_avg);
      EXXXX(N, a, b, cds->NSfields, cds->o_S, o_S4_avg);
    }

  }

//////////TKE BUDGET//////////

  if(comp_TKEbudget) {

    const dlong N = mesh->Nelements * mesh->Np;

    nrs->gradientVolumeKernel(
    mesh->Nelements,
    mesh->o_vgeo,
    mesh->o_D,
    nrs->fieldOffset,
    o_vx,
    o_dvx);

    oogs::startFinish(o_dvx, nrs->NVfields, nrs->fieldOffset,ogsDfloat, ogsAdd, nrs->gsh);

    platform->linAlg->axmyVector(
      mesh->Nlocal,
      nrs->fieldOffset,
      0,
      1.0,
      nrs->meshV->o_invLMM,
      o_dvx);

    nrs->gradientVolumeKernel(
    mesh->Nelements,
    mesh->o_vgeo,
    mesh->o_D,
    nrs->fieldOffset,
    o_vy,
    o_dvy);

    oogs::startFinish(o_dvy, nrs->NVfields, nrs->fieldOffset,ogsDfloat, ogsAdd, nrs->gsh);

    platform->linAlg->axmyVector(
      mesh->Nlocal,
      nrs->fieldOffset,
      0,
      1.0,
      nrs->meshV->o_invLMM,
      o_dvy);

    nrs->gradientVolumeKernel(
    mesh->Nelements,
    mesh->o_vgeo,
    mesh->o_D,
    nrs->fieldOffset,
    o_vz,
    o_dvz);

    oogs::startFinish(o_dvz, nrs->NVfields, nrs->fieldOffset,ogsDfloat, ogsAdd, nrs->gsh);

    platform->linAlg->axmyVector(
      mesh->Nlocal,
      nrs->fieldOffset,
      0,
      1.0,
      nrs->meshV->o_invLMM,
      o_dvz);

    EXX(N, a, b, nrs->NVfields, o_dvx, o_dxu2_avg);
    EXX(N, a, b, nrs->NVfields, o_dvy, o_dxv2_avg);
    EXX(N, a, b, nrs->NVfields, o_dvz, o_dxw2_avg);

    EXXYsame(N, a, b, 1, o_vx, o_vy, o_u2U_avg + 0 * offsetByte);
    EXXYsame(N, a, b, 1, o_vx, o_vz, o_u2U_avg + 1 * offsetByte);
    o_u2U_avg.copyFrom(o_U_avg + 0 * offsetByte, mesh->Nlocal*sizeof(dfloat), 2*offsetByte);
    EXXYsame(N, a, b, 1, o_vy, o_vx, o_v2U_avg + 0 * offsetByte);
    EXXYsame(N, a, b, 1, o_vy, o_vz, o_v2U_avg + 1 * offsetByte);
    o_v2U_avg.copyFrom(o_U_avg + 1 * offsetByte, mesh->Nlocal*sizeof(dfloat), 2*offsetByte);
    EXXYsame(N, a, b, 1, o_vz, o_vx, o_w2U_avg + 0 * offsetByte);
    EXXYsame(N, a, b, 1, o_vz, o_vy, o_w2U_avg + 1 * offsetByte);
    o_w2U_avg.copyFrom(o_U_avg + 2 * offsetByte, mesh->Nlocal*sizeof(dfloat), 2*offsetByte);

    EXYsame(N, a, b, 1, nrs->o_P, o_dvx + 0 * offsetByte, o_PdxU_avg + 0 * offsetByte);
    EXYsame(N, a, b, 1, nrs->o_P, o_dvy + 1 * offsetByte, o_PdxU_avg + 1 * offsetByte);
    EXYsame(N, a, b, 1, nrs->o_P, o_dvz + 2 * offsetByte, o_PdxU_avg + 2 * offsetByte);

    EXYdiff(N, a, b, nrs->NVfields, 1, nrs->o_U, nrs->o_P, o_PU_avg);

   }

//////////THF BUDGET//////////

  if(comp_THFbudget) {

    cds_t* cds = nrs->cds;

    const dlong N = mesh->Nelements * mesh->Np;

    o_s0 = cds->o_S + 0 * offsetByte;
    o_s1 = cds->o_S + 1 * offsetByte;

    nrs->gradientVolumeKernel(
    mesh->Nelements,
    mesh->o_vgeo,
    mesh->o_D,
    nrs->fieldOffset,
    o_s0,
    o_ds0);
    
    oogs::startFinish(o_ds0, nrs->NVfields, nrs->fieldOffset,ogsDfloat, ogsAdd, nrs->gsh);

    platform->linAlg->axmyVector(
      mesh->Nlocal,
      nrs->fieldOffset,
      0,
      1.0,
      nrs->meshV->o_invLMM,
      o_ds0);

    nrs->gradientVolumeKernel(
    mesh->Nelements,
    mesh->o_vgeo,
    mesh->o_D,
    nrs->fieldOffset,
    o_s1,
    o_ds1);

    oogs::startFinish(o_ds1, nrs->NVfields, nrs->fieldOffset,ogsDfloat, ogsAdd, nrs->gsh);

    platform->linAlg->axmyVector(
      mesh->Nlocal,
      nrs->fieldOffset,
      0,
      1.0,
      nrs->meshV->o_invLMM,
      o_ds1);

    EXYsame(N, a, b, nrs->NVfields, o_dvx, o_ds0, o_dxuS_avg + 0 * nrs->NVfields * offsetByte);
    EXYsame(N, a, b, nrs->NVfields, o_dvx, o_ds1, o_dxuS_avg + 1 * nrs->NVfields * offsetByte);
    EXYsame(N, a, b, nrs->NVfields, o_dvy, o_ds0, o_dxvS_avg + 0 * nrs->NVfields * offsetByte);
    EXYsame(N, a, b, nrs->NVfields, o_dvy, o_ds1, o_dxvS_avg + 1 * nrs->NVfields * offsetByte);
    EXYsame(N, a, b, nrs->NVfields, o_dvz, o_ds0, o_dxwS_avg + 0 * nrs->NVfields * offsetByte);
    EXYsame(N, a, b, nrs->NVfields, o_dvz, o_ds1, o_dxwS_avg + 1 * nrs->NVfields * offsetByte);

    EXYdiff(N, a, b, nrs->NVfields, cds->NSfields, o_dvx, cds->o_S, o_Sdxu_avg);
    EXYdiff(N, a, b, nrs->NVfields, cds->NSfields, o_dvy, cds->o_S, o_Sdxv_avg);
    EXYdiff(N, a, b, nrs->NVfields, cds->NSfields, o_dvz, cds->o_S, o_Sdxw_avg);

    EXYdiff(N, a, b, nrs->NVfields, 1, o_ds0, o_vx, o_udxS_avg + 0 * nrs->NVfields * offsetByte);
    EXYdiff(N, a, b, nrs->NVfields, 1, o_ds1, o_vx, o_udxS_avg + 1 * nrs->NVfields * offsetByte);
    EXYdiff(N, a, b, nrs->NVfields, 1, o_ds0, o_vy, o_vdxS_avg + 0 * nrs->NVfields * offsetByte);
    EXYdiff(N, a, b, nrs->NVfields, 1, o_ds1, o_vy, o_vdxS_avg + 1 * nrs->NVfields * offsetByte);
    EXYdiff(N, a, b, nrs->NVfields, 1, o_ds0, o_vz, o_wdxS_avg + 0 * nrs->NVfields * offsetByte);
    EXYdiff(N, a, b, nrs->NVfields, 1, o_ds1, o_vz, o_wdxS_avg + 1 * nrs->NVfields * offsetByte);

    EXYdiff(N, a, b, nrs->NVfields, 1, o_ds0, nrs->o_P, o_PdxS_avg + 0 * nrs->NVfields * offsetByte);
    EXYdiff(N, a, b, nrs->NVfields, 1, o_ds1, nrs->o_P, o_PdxS_avg + 1 * nrs->NVfields * offsetByte);

    EXXYdiff(N, a, b, nrs->NVfields, cds->NSfields, nrs->o_U, cds->o_S, o_UUS_avg);

// ARK USED the 2 commented lines below for the budgets...BUT not really for comparison yet???
    EXXYdiff(N, a, b, 1, nrs->NVfields, o_s0, nrs->o_U, o_SSU_avg + 0 * nrs->NVfields * offsetByte);
    EXXYdiff(N, a, b, 1, nrs->NVfields, o_s1, nrs->o_U, o_SSU_avg + 1 * nrs->NVfields * offsetByte);

// ARK NOTE that in new version SHOULD change the order to uv, vw, wu instead
    EXYZsame(N, a, b, 1, 1, o_vx, o_vy, o_s0, o_uvS_avg + 0 * offsetByte);
    EXYZsame(N, a, b, 1, 1, o_vx, o_vz, o_s0, o_uvS_avg + 1 * offsetByte);
    EXYZsame(N, a, b, 1, 1, o_vy, o_vz, o_s0, o_uvS_avg + 2 * offsetByte);
    EXYZsame(N, a, b, 1, 1, o_vx, o_vy, o_s1, o_uvS_avg + 3 * offsetByte);
    EXYZsame(N, a, b, 1, 1, o_vx, o_vz, o_s1, o_uvS_avg + 4 * offsetByte);
    EXYZsame(N, a, b, 1, 1, o_vy, o_vz, o_s1, o_uvS_avg + 5 * offsetByte);

  } 

  timel = time;
}

void budgets::setup(nrs_t* nrs_, bool comp_skew, bool comp_TKEbudget, bool comp_THFbudget)
{
  nrsCheck(!buildKernelCalled, MPI_COMM_SELF, EXIT_FAILURE,
         "%s\n", "called prior tavg::buildKernel()!");
  nrs = nrs_;
  mesh_t* mesh = nrs->meshV;
  
  if(setupCalled) return;

  o_U_avg = platform->device.malloc(nrs->fieldOffset * nrs->NVfields ,  sizeof(dfloat));
  o_U2_avg = platform->device.malloc(nrs->fieldOffset * nrs->NVfields ,  sizeof(dfloat));
  platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields, 0.0, o_U_avg);
  platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields, 0.0, o_U2_avg);

  o_u_cov = platform->device.malloc(nrs->fieldOffset * nrs->NVfields ,  sizeof(dfloat));
  o_v_cov = platform->device.malloc(nrs->fieldOffset * nrs->NVfields ,  sizeof(dfloat));
  o_w_cov = platform->device.malloc(nrs->fieldOffset * nrs->NVfields ,  sizeof(dfloat));
  platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields, 0.0, o_u_cov);
  platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields, 0.0, o_v_cov);
  platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields, 0.0, o_w_cov);
  o_vx = platform->device.malloc(nrs->fieldOffset ,  sizeof(dfloat));
  o_vy = platform->device.malloc(nrs->fieldOffset ,  sizeof(dfloat));
  o_vz = platform->device.malloc(nrs->fieldOffset ,  sizeof(dfloat));
  o_uu_avg = platform->device.malloc(nrs->fieldOffset ,  sizeof(dfloat));
  o_vv_avg = platform->device.malloc(nrs->fieldOffset ,  sizeof(dfloat));
  o_ww_avg = platform->device.malloc(nrs->fieldOffset ,  sizeof(dfloat));
  platform->linAlg->fill(nrs->fieldOffset, 0.0, o_vx);
  platform->linAlg->fill(nrs->fieldOffset, 0.0, o_vy);
  platform->linAlg->fill(nrs->fieldOffset, 0.0, o_vz);
  platform->linAlg->fill(nrs->fieldOffset, 0.0, o_uu_avg);
  platform->linAlg->fill(nrs->fieldOffset, 0.0, o_vv_avg);
  platform->linAlg->fill(nrs->fieldOffset, 0.0, o_ww_avg);

  o_P_avg = platform->device.malloc(nrs->fieldOffset ,  sizeof(dfloat));
  o_P2_avg = platform->device.malloc(nrs->fieldOffset ,  sizeof(dfloat));
  platform->linAlg->fill(nrs->fieldOffset, 0.0, o_P_avg);
  platform->linAlg->fill(nrs->fieldOffset, 0.0, o_P2_avg);

  if(nrs->Nscalar) {
    cds_t* cds = nrs->cds;
//ARK why no filling w/ 0 here?
    o_S_avg = platform->device.malloc(cds->fieldOffsetSum,  sizeof(dfloat));
    o_S2_avg = platform->device.malloc(cds->fieldOffsetSum,  sizeof(dfloat));
    o_uS_avg = platform->device.malloc(cds->fieldOffsetSum,  sizeof(dfloat));
    o_vS_avg = platform->device.malloc(cds->fieldOffsetSum,  sizeof(dfloat));
    o_wS_avg = platform->device.malloc(cds->fieldOffsetSum,  sizeof(dfloat));
    o_PS_avg = platform->device.malloc(cds->fieldOffsetSum,  sizeof(dfloat));
  }

  if(comp_skew) {

    o_U3_avg = platform->device.malloc(nrs->fieldOffset * nrs->NVfields ,  sizeof(dfloat));
    o_U4_avg = platform->device.malloc(nrs->fieldOffset * nrs->NVfields ,  sizeof(dfloat));
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields, 0.0, o_U3_avg);
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields, 0.0, o_U4_avg);
  
    o_P3_avg = platform->device.malloc(nrs->fieldOffset ,  sizeof(dfloat));
    o_P4_avg = platform->device.malloc(nrs->fieldOffset ,  sizeof(dfloat));
    platform->linAlg->fill(nrs->fieldOffset, 0.0, o_P3_avg);
    platform->linAlg->fill(nrs->fieldOffset, 0.0, o_P4_avg);

    if(nrs->Nscalar) {
      cds_t* cds = nrs->cds;
      o_S3_avg = platform->device.malloc(cds->fieldOffsetSum,  sizeof(dfloat));
      o_S4_avg = platform->device.malloc(cds->fieldOffsetSum,  sizeof(dfloat));
    }
  
  }

  if(comp_TKEbudget || comp_THFbudget) {

    o_dvx = platform->device.malloc(nrs->fieldOffset * nrs->NVfields ,  sizeof(dfloat));
    o_dvy = platform->device.malloc(nrs->fieldOffset * nrs->NVfields ,  sizeof(dfloat));
    o_dvz = platform->device.malloc(nrs->fieldOffset * nrs->NVfields ,  sizeof(dfloat));
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields, 0.0, o_dvx);
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields, 0.0, o_dvy);
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields, 0.0, o_dvz);

  }

  if(comp_TKEbudget) {
 
    //ARK assumes that NVfields is the dimensionality of the problem...check if there's a better variable
    //    prior stuff is done above assuming vx/vy/vz exist anyway...
    o_dxu2_avg = platform->device.malloc(nrs->fieldOffset * nrs->NVfields ,  sizeof(dfloat));
    o_dxv2_avg = platform->device.malloc(nrs->fieldOffset * nrs->NVfields ,  sizeof(dfloat));
    o_dxw2_avg = platform->device.malloc(nrs->fieldOffset * nrs->NVfields ,  sizeof(dfloat));
    o_PdxU_avg = platform->device.malloc(nrs->fieldOffset * nrs->NVfields ,  sizeof(dfloat));
    o_u2U_avg  = platform->device.malloc(nrs->fieldOffset * nrs->NVfields ,  sizeof(dfloat));
    o_v2U_avg  = platform->device.malloc(nrs->fieldOffset * nrs->NVfields ,  sizeof(dfloat));
    o_w2U_avg  = platform->device.malloc(nrs->fieldOffset * nrs->NVfields ,  sizeof(dfloat));
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields, 0.0, o_dxu2_avg);
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields, 0.0, o_dxv2_avg);
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields, 0.0, o_dxw2_avg);
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields, 0.0, o_PdxU_avg);
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields, 0.0, o_u2U_avg);
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields, 0.0, o_v2U_avg);
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields, 0.0, o_w2U_avg);

    o_PU_avg = platform->device.malloc(nrs->fieldOffset * nrs->NVfields , sizeof(dfloat));
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields, 0.0, o_PU_avg);

  }

  if(comp_THFbudget) {

    cds_t* cds = nrs->cds;

    //ARK note that this is also outputting the few extra terms which are used in the temperature variance budget,
    //    namely ttu, ttv, ttw
    o_s0 = platform->device.malloc(nrs->fieldOffset ,  sizeof(dfloat));
    o_s1 = platform->device.malloc(nrs->fieldOffset ,  sizeof(dfloat));
    o_ds0 = platform->device.malloc(nrs->fieldOffset * nrs->NVfields ,  sizeof(dfloat));
    o_ds1 = platform->device.malloc(nrs->fieldOffset * nrs->NVfields ,  sizeof(dfloat));
    platform->linAlg->fill(nrs->fieldOffset, 0.0, o_s0);
    platform->linAlg->fill(nrs->fieldOffset, 0.0, o_s1);
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields, 0.0, o_ds0);
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields, 0.0, o_ds1);

    o_dxuS_avg = platform->device.malloc(nrs->fieldOffset * nrs->NVfields * cds->NSfields ,  sizeof(dfloat));
    o_dxvS_avg = platform->device.malloc(nrs->fieldOffset * nrs->NVfields * cds->NSfields ,  sizeof(dfloat));
    o_dxwS_avg = platform->device.malloc(nrs->fieldOffset * nrs->NVfields * cds->NSfields ,  sizeof(dfloat));
    o_Sdxu_avg = platform->device.malloc(nrs->fieldOffset * nrs->NVfields * cds->NSfields ,  sizeof(dfloat));
    o_Sdxv_avg = platform->device.malloc(nrs->fieldOffset * nrs->NVfields * cds->NSfields ,  sizeof(dfloat));
    o_Sdxw_avg = platform->device.malloc(nrs->fieldOffset * nrs->NVfields * cds->NSfields ,  sizeof(dfloat));
    o_udxS_avg = platform->device.malloc(nrs->fieldOffset * nrs->NVfields * cds->NSfields ,  sizeof(dfloat));
    o_vdxS_avg = platform->device.malloc(nrs->fieldOffset * nrs->NVfields * cds->NSfields ,  sizeof(dfloat));
    o_wdxS_avg = platform->device.malloc(nrs->fieldOffset * nrs->NVfields * cds->NSfields ,  sizeof(dfloat));
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields * cds->NSfields, 0.0, o_dxuS_avg);
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields * cds->NSfields, 0.0, o_dxvS_avg);
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields * cds->NSfields, 0.0, o_dxwS_avg);
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields * cds->NSfields, 0.0, o_Sdxu_avg);
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields * cds->NSfields, 0.0, o_Sdxv_avg);
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields * cds->NSfields, 0.0, o_Sdxw_avg);
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields * cds->NSfields, 0.0, o_udxS_avg);
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields * cds->NSfields, 0.0, o_vdxS_avg);
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields * cds->NSfields, 0.0, o_wdxS_avg);

    o_UUS_avg = platform->device.malloc(nrs->fieldOffset * nrs->NVfields * cds->NSfields ,  sizeof(dfloat));
    o_SSU_avg = platform->device.malloc(nrs->fieldOffset * nrs->NVfields * cds->NSfields ,  sizeof(dfloat));
    o_uvS_avg = platform->device.malloc(nrs->fieldOffset * nrs->NVfields * cds->NSfields ,  sizeof(dfloat));
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields * cds->NSfields, 0.0, o_UUS_avg);
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields * cds->NSfields, 0.0, o_SSU_avg);
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields * cds->NSfields, 0.0, o_uvS_avg);

    o_PdxS_avg = platform->device.malloc(nrs->fieldOffset * nrs->NVfields * cds->NSfields ,  sizeof(dfloat));
    platform->linAlg->fill(nrs->fieldOffset * nrs->NVfields * cds->NSfields, 0.0, o_PdxS_avg);

  }

  setupCalled = 1;
}

void budgets::outfld(int _outXYZ, int FP64, bool comp_skew, bool comp_TKEbudget, bool comp_THFbudget)
{
  if(!nrs->timeStepConverged) return;

  cds_t* cds = nrs->cds;
  mesh_t* mesh = nrs->meshV;

  const dlong offsetByte = nrs->fieldOffset * sizeof(dfloat);

  int outXYZ = _outXYZ;
  if(!outfldCounter) outXYZ = 1;

  occa::memory o_null;
  occa::memory o_T_avg, o_T2_avg;
  occa::memory o_T3_avg, o_T4_avg;
  occa::memory o_uT_avg, o_vT_avg, o_wT_avg;
  occa::memory o_dxuT_avg, o_dxvT_avg, o_dxwT_avg;
  occa::memory o_Tdxu_avg, o_Tdxv_avg, o_Tdxw_avg;
  occa::memory o_Tdyu_avg, o_Tdyv_avg, o_Tdyw_avg;
  occa::memory o_Tdzu_avg, o_Tdzv_avg, o_Tdzw_avg;
  occa::memory o_udxT_avg, o_vdxT_avg, o_wdxT_avg;
  occa::memory o_UUT_avg;
  occa::memory o_TTU_avg;
  occa::memory o_uvT_avg;

  const int Nscalar = nrs->Nscalar;
  if(nrs->Nscalar) {
    o_T_avg = o_S_avg;
    o_T2_avg = o_S2_avg;
    if(comp_skew) {
      o_T3_avg = o_S3_avg;
      o_T4_avg = o_S4_avg;
    }

    o_uT_avg = o_uS_avg;
    o_vT_avg = o_vS_avg;
    o_wT_avg = o_wS_avg;

  }

  writeFld("ra1", atime, outfldCounter, outXYZ, FP64,
           &o_U_avg,
           &o_P_avg,
           &o_T_avg,
           Nscalar);

  writeFld("ra2", atime, outfldCounter, outXYZ, FP64,
           &o_U2_avg,
           &o_P2_avg,
           &o_T2_avg,
           Nscalar);

  if (comp_skew) {

    writeFld("ra3", atime, outfldCounter, outXYZ, FP64,
             &o_U3_avg,
             &o_P3_avg,
             &o_T3_avg,
             Nscalar);

    writeFld("ra4", atime, outfldCounter, outXYZ, FP64,
             &o_U4_avg,
             &o_P4_avg,
             &o_T4_avg,
             Nscalar);

  } //if comp_skew

  writeFld("ra5", atime, outfldCounter, outXYZ, FP64,
           &o_u_cov,
           &o_P_avg,
           &o_uT_avg,
           Nscalar);

  writeFld("ra6", atime, outfldCounter, outXYZ, FP64,
           &o_v_cov,
           &o_P_avg,
           &o_vT_avg,
           Nscalar);

  writeFld("ra7", atime, outfldCounter, outXYZ, FP64,
           &o_w_cov,
           &o_P_avg,
           &o_wT_avg,
           Nscalar);

  if (comp_TKEbudget) {
  
  writeFld("rk1", atime, outfldCounter, outXYZ, FP64,
           &o_dxu2_avg,
           &o_P_avg,
           &o_T_avg,
           Nscalar);

  writeFld("rk2", atime, outfldCounter, outXYZ, FP64,
           &o_dxv2_avg,
           &o_P_avg,
           &o_T_avg,
           Nscalar);

  writeFld("rk3", atime, outfldCounter, outXYZ, FP64,
           &o_dxw2_avg,
           &o_P_avg,
           &o_T_avg,
           Nscalar);

  writeFld("rk4", atime, outfldCounter, outXYZ, FP64,
           &o_u2U_avg,
           &o_P_avg,
           &o_T_avg,
           Nscalar);

  writeFld("rk5", atime, outfldCounter, outXYZ, FP64,
           &o_v2U_avg,
           &o_P_avg,
           &o_T_avg,
           Nscalar);

  writeFld("rk6", atime, outfldCounter, outXYZ, FP64,
           &o_w2U_avg,
           &o_P_avg,
           &o_T_avg,
           Nscalar);

  writeFld("rk7", atime, outfldCounter, outXYZ, FP64,
           &o_PdxU_avg,
           &o_P_avg,
           &o_T_avg,
           Nscalar);

  writeFld("rk8", atime, outfldCounter, outXYZ, FP64,
           &o_PU_avg,
           &o_P_avg,
           &o_T_avg,
           Nscalar);

  } // if comp_TKEbudget

  if (comp_THFbudget) {

    for (int i=0; i < nrs->Nscalar; i++) {
      auto j = std::to_string(i+1);

      occa::memory o_dxuT_avg = o_dxuS_avg + i * nrs->NVfields * offsetByte;
      occa::memory o_dxvT_avg = o_dxvS_avg + i * nrs->NVfields * offsetByte;
      occa::memory o_dxwT_avg = o_dxwS_avg + i * nrs->NVfields * offsetByte;
      occa::memory o_Tdxu_avg = o_Sdxu_avg + 0 * offsetByte + i * nrs->NVfields * offsetByte;
      occa::memory o_Tdxv_avg = o_Sdxv_avg + 0 * offsetByte + i * nrs->NVfields * offsetByte;
      occa::memory o_Tdxw_avg = o_Sdxw_avg + 0 * offsetByte + i * nrs->NVfields * offsetByte;
      occa::memory o_Tdyu_avg = o_Sdxu_avg + 1 * offsetByte + i * nrs->NVfields * offsetByte;
      occa::memory o_Tdyv_avg = o_Sdxv_avg + 1 * offsetByte + i * nrs->NVfields * offsetByte;
      occa::memory o_Tdyw_avg = o_Sdxw_avg + 1 * offsetByte + i * nrs->NVfields * offsetByte;
      occa::memory o_Tdzu_avg = o_Sdxu_avg + 2 * offsetByte + i * nrs->NVfields * offsetByte;
      occa::memory o_Tdzv_avg = o_Sdxv_avg + 2 * offsetByte + i * nrs->NVfields * offsetByte;
      occa::memory o_Tdzw_avg = o_Sdxw_avg + 2 * offsetByte + i * nrs->NVfields * offsetByte;
      occa::memory o_udxT_avg = o_udxS_avg + i * nrs->NVfields * offsetByte;
      occa::memory o_vdxT_avg = o_vdxS_avg + i * nrs->NVfields * offsetByte;
      occa::memory o_wdxT_avg = o_wdxS_avg + i * nrs->NVfields * offsetByte;
      occa::memory o_PdxT_avg = o_PdxS_avg + i * nrs->NVfields * offsetByte;
      occa::memory o_UUT_avg = o_UUS_avg + i * nrs->NVfields * offsetByte;
      occa::memory o_TTU_avg = o_SSU_avg + i * nrs->NVfields * offsetByte;
      occa::memory o_uvT_avg = o_uvS_avg + i * nrs->NVfields * offsetByte;

      writeFld("r" + j + "0", atime, outfldCounter, outXYZ, FP64,
               &o_dxuT_avg,
               &o_Tdxu_avg,
               &o_T_avg,
               Nscalar);

      writeFld("r" + j + "1", atime, outfldCounter, outXYZ, FP64,
               &o_dxvT_avg,
               &o_Tdxv_avg,
               &o_T_avg,
               Nscalar);

      writeFld("r" + j + "2", atime, outfldCounter, outXYZ, FP64,
               &o_dxwT_avg,
               &o_Tdxw_avg,
               &o_T_avg,
               Nscalar);

      writeFld("r" + j + "3", atime, outfldCounter, outXYZ, FP64,
               &o_udxT_avg,
               &o_Tdyu_avg,
               &o_T_avg,
               Nscalar);

      writeFld("r" + j + "4", atime, outfldCounter, outXYZ, FP64,
               &o_vdxT_avg,
               &o_Tdyv_avg,
               &o_T_avg,
               Nscalar);

      writeFld("r" + j + "5", atime, outfldCounter, outXYZ, FP64,
               &o_wdxT_avg,
               &o_Tdyw_avg,
               &o_T_avg,
               Nscalar);

      writeFld("r" + j + "6", atime, outfldCounter, outXYZ, FP64,
               &o_PdxT_avg,
               &o_Tdzu_avg,
               &o_T_avg,
               Nscalar);

      writeFld("r" + j + "7", atime, outfldCounter, outXYZ, FP64,
               &o_UUT_avg,
               &o_Tdzv_avg,
               &o_T_avg,
               Nscalar);

      writeFld("r" + j + "8", atime, outfldCounter, outXYZ, FP64,
               &o_uvT_avg,
               &o_Tdzw_avg,
               &o_PS_avg,
               Nscalar);

      writeFld("r" + j + "9", atime, outfldCounter, outXYZ, FP64,
               &o_TTU_avg,
               &o_P_avg,
               &o_T_avg,
               Nscalar);

    }
  } // if comp_THFbudget

  atime = 0;
  outfldCounter++;
}


void budgets::outfld(bool comp_skew, bool comp_TKEbudget, bool comp_THFbudget)
{
  budgets::outfld(/* outXYZ */ 0, /* FP64 */ 1, comp_skew, comp_TKEbudget, comp_THFbudget);
}
