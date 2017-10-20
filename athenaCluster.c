#include "copyright.h"
/*============================================================================*/
/*! \file athenaGM07.c
 *  \brief Problem generator for GM2007 Chemical Network.3D
 *
 * PURPOSE: Problem generator for Chemical Network GM problem (No Turbulence)
 *
 * REFERENCE: Glover MacLow 2007.     */
/*============================================================================*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "defs.h"
#include "athena.h"
#include "globals.h"
#include "prototypes.h"

//CONSTANTS
#define MU_H 1.67E-24  //g
#define KB 1.38E-16    //ergs K-1
#define MYR 3.14E13    //s Myr-1
#define PCCM 3.08E18   //cm pc-1

/*==============================================================================
 * PROBLEM PRIVATE USER FUNCTIONS:
 * Prob_Init()        - Sets up Initial Conditions and Interp table arrays
 * INTERPOLATION
 * GetInterpolation() - Performs 3D Linear Interpolation for eqmTemparr
 * HELPER FUNCTIONS
 * GetAbund()         - Returns Chemical Species Abundances
 * GetnH2()           - Finds value of H2 number density given H2 abundance
 * GetnHp()           - Finds value of H+ number density given H+ abundance
 * GetnH()            - Finds value of H number density given H abundance
 * Getne()            - Finds value of e number density given e abundance
 * GetxH2()           - Finds value of H2 abundance given H2 number density
 * GetxHp()           - Finds value of H+ abundance given H+ number density
 * GetxH()            - Finds value of H abundance given H+ and H2 number densities
 * Getxe()            - Finds value of e abundance given H+ abundance
 * CHEMICAL NETWORK FUNCTIONS
 * RunCN()            - Performs Chemical Network Simulation
 * dnxHdt()           - Finds the rate of change of our Hydrogen species vs time
 * dnH2()             - Finds the rate of change for H2 vs time
 * dnHp()             - Finds the rate of change for H+ vs time
 * k1()               - H2 Formation from grains
 * k2()               - H2 collisional dissociation with H
 * k3()               - H2 collisional dissociation with H2
 * k4()               - H2 Photodissociation 
 * k5()               - H+ Formation (Cosmic Ray Ionization)
 * k6()               - H+ collisional formation with e
 * k7()               - H+ collisional dissociation with e
 * k8()               - H+ dissociation from grains
 * THERMAL MODEL FUNCTIONS
 * RunThermal()       - Performs Thermal Model Simulation
 * dTdt()             - Finds the rate of change of Temperature vs time
 * GetNetCool()       - Finds the net rate of heating/cooling in ergs s-1
 * GetLoss()          - Finds the net rate of cooling in ergs s-1
 * GetGain()          - Finds the net rate of heating in ergs s-1
 * spline()           - Cubic spline interpolation method
 * splint()           - Cubic spline interpolation method
 * Getsigv()          - Returns collisional rate in cm3 s-1
 * INTEGRATION FUNCTIONS
 * odeint()           - Calculates new values of y after a defined change in x
 * stiff()            - 4th order Rosenbrock step for integrating stiff ODEs (adaptive step size)
 * RKQS()             - Function implementing 5th order RK step
 * SELF GRAVITY FUNCTIONS
 * GetMeanRho()       - Calculates global variable grav_mean_rho
 *-----------------------------------------------------------------------------*/
//ERROR CHECK
static void CheckAbundance();
static void CheckEnergy(double energyValue,double KEvalue);
//INTERPOLATION
static double GetInterpolation(double powDens, double powxH2, double powxHp, double *eqmTemparr);
//HELPER FUNCTIONS
static void GetAbund(double abund[]);
static double GetnH2(double xH2);
static double GetnHp(double xHp);
static double Getne(double nHp);
static double GetxH2(double nH2);
static double GetxHp(double nHp);
static double GetxH(double nH2, double nHp);
static double Getxe(double nHp);
//CHEMICAL NETWORK FUNCTIONS
static void RunCN(double tstart, double tend, double vecnHx[]);
static void dnHxdt(double valt, int nvar, double arrnHx[], double dnHxdt[]);
static double dnH2(double nHx[]);
static double dnHp(double nHx[]);
static double k1(double temp, double tempgrain);
static double k2(double temp, double nH2, double nHp);
static double k3(double temp, double nH2, double nHp);
static double k4();
static double k4ss(); //local self shielding
static double k5();
static double k6(double temp);
static double k7(double temp);
static double k8(double temp, double ne);
//MULTI ROOT FINDER
static void MNRootInit(double nHxmid[]);
static void MultiNewt(int ntrial, double x[], int n, double tolx, double tolf);
//THERMAL MODEL FUNCTIONS
static void RunThermal(double tstart, double tend, double vecTemp[]);
static void dTdt(double valt, int nvar, double arrTemp[], double dTdt[]);
static double GetNetCool(double temp);
static double GetLoss(double T, double xe, double xO);
static double GetGain(double T, double xe);
static void spline(double x[], double y[], int n, double yp1, double ypn, double y2[]);
static void splint(double xa[], double ya[], double y2a[], int n, double x, double *y);
static double Getsigv(double Temp, double cs, double degen);
//INTEGRATION FUNCTIONS
static void odeint(double ystart[], int nvar, double x1, double x2, double eps,
		   double h1, double hmin, int *nok, int *nbad, int boolCN,
		   void (*derivs)(double, int, double[], double[]),
		   void (*rkqs)(double[], double[], int, double*, double, double, double[],
				double*, double*, void (*)(double, int, double[], double[])));
static void stiff(double y[], double dydx[], int n, double *x, double htry, double eps,
		  double yscal[], double *hdid, double *hnext, 
		  void (*derivs)(double, int, double[], double[]));
static void RKQS(double y[], double dydx[], int n, double *x, double htry, double eps,
		  double yscal[], double *hdid, double *hnext, 
		 void (*derivs)(double, int, double[], double[]));
static double GetRoot(double Tmid, double time);
//SELF GRAVITY FUNCTION
#ifdef SELF_GRAVITY
static void GetMeanRho(DomainS *pDomain);
#endif /*SELF_GRAVITY*/
/*==============================================================================
 * PROBLEM PRIVATE USER HELPER FUNCTIONS
 * vector             - Creates array of size(1,n)
 * ivector            - Creates int array of size(1,n)
 * matrix             - Creates matrix of size(n1,n2) index starts at 1
 * free_vector        - Frees allocation of vector
 * free_ivector       - Frees allocation of ivector
 * free_matrix        - Frees allocation of matrix
 * nrerror            - Prints error message and closes application
 *-----------------------------------------------------------------------------*/
static double *vector(long nl, long nh);
static int *ivector(long nl, long nh);
static double **matrix(long nrl, long nrh, long ncl, long nch);
static void free_vector(double *v, long nl, long nh);
static void free_ivector(int *v, long nl, long nh);
static void free_matrix(double **m, long nrl, long nrh, long ncl, long nch);
static void nrerror(char error_text[]);

//STATIC VARIABLES AND DEFINITIONS

//PROBLEM GENERATION
//Athinput Variables
static double valxH2, valxHp, valDens, valsV, valTemp;
static double hvalueCN, hvalueT;
static int Nx1, Nx2, Nx3;
static double x1max, x2max, x3max;
static double abund[5];
#define aH 0
#define aHe 1
#define aO 2
#define aC 3
#define aSi 4
//Cluster Generation
static int GNx1, GNx2, GNx3;
static double lx1, lx2, lx3;
static double *deltaLogDens;
static double globalmindld = 0.0, globalmaxdld = 0.0;
//Temperature Interpolation
static int dim1, dim2, dim3;
static double *powxH2arr, *powxHparr, *powDensarr, *eqmTemparr;
#define ARReqmTemp(i,j,k) (eqmTemparr[dim2*dim3*i + dim3*j + k])
static double minxH2, minxHp, minDens;
static double maxxH2, maxxHp, maxDens;
//USERWORK IN LOOP
static double xpos, ypos, zpos;
static double *vecnHx, *vecTemp;
static double globalMaxH2 = 0.0;
static double globalMaxDens = 0.0;
#define NVARCN 2
#define NVART 1
static double tstart, tend;
static double b4xH2, b4xHp, b4Temp;
static double a4xH2, a4xHp, a4Temp;
static double eqmTemp, eqmxH2, eqmxHp;
#define TOL 1.0E-2
static int boolCN, boolTM;
//Chemical Network
#define KMAX 100
#define HMIN 0.0
#define TINY 1.0E-30
#define MAXSTPS 10000
#define EPS 1.0E-5
static double dxsav;
static int kmax,kount;
static double *xp, **yp;
//dnHxdt
#define CHI 1.7
#define GRAIN_TEMP 65.0
#define ZETA 1.8E-17
//THERMAL MODEL STATICS AND CONSTANTS
#define HPLANCK 6.67E-27 //ergs s
#define CSPEED 3.0E10 //cm s-1

//PRIVATE USER HELPER FUNCTIONS STATIC AND CONSTANTS
#define NR_END 1
#define FREE_ARG char*
#define SGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
static double maxarg1, maxarg2;
#define FMAX(a,b) (maxarg1=(a),maxarg2=(b),(maxarg1) > (maxarg2) ? (maxarg1) : (maxarg2))
static double minarg1,minarg2;
#define FMIN(a,b) (minarg1=(a),minarg2=(b),(minarg1) < (minarg2) ? (minarg1) : (minarg2))

void problem(DomainS *pDomain)
{
  //Grid Parameters
  GridS *pGrid=(pDomain->Grid);
  Prim1DS W;
  Cons1DS U1d;
  int i, is = pGrid->is, ie = pGrid->ie;
  int j, js = pGrid->js, je = pGrid->je;
  int k, ks = pGrid->ks, ke = pGrid->ke;
  //Hydro Parameters
  Real press,rhotot,x1,x2,x3;
  Real KE;
  Real b0=0.0,Bx=0.0;
  double theta;
  //Additional Initial Condition Parameters
  double rad,dist,newTemp;


  //Create Abundance Array
  GetAbund(abund);

  //Initial Conditions
  four_pi_G     = par_getd("problem","four_pi_G");
  grav_mean_rho = par_getd("problem","grav_mean_rho");
  valDens       = par_getd("problem","numdenstot");
  valTemp       = par_getd("problem","temp");
  valxH2        = par_getd("problem","xH2");
  valxHp        = par_getd("problem","xHp");
  hvalueCN      = par_getd("problem","hCN");
  hvalueT       = par_getd("problem","hT");
  boolCN        = par_getd("problem","boolCN");
  boolTM        = par_getd("problem","boolTM");

  Nx1           = par_getd("domain1","Nx1");
  Nx2           = par_getd("domain1","Nx2");
  Nx3           = par_getd("domain1","Nx3");
  x1max         = par_getd("domain1","x1max");
  x2max         = par_getd("domain1","x2max");
  x3max         = par_getd("domain1","x3max");

  //pGrid # of cells
  GNx1 = pGrid->Nx[0];
  GNx2 = pGrid->Nx[1];
  GNx3 = pGrid->Nx[2];

  //Length of grid in simulation
  lx1 = pDomain->RootMaxX[0] - pDomain->RootMinX[0];
  lx2 = pDomain->RootMaxX[1] - pDomain->RootMinX[1];
  lx3 = pDomain->RootMaxX[2] - pDomain->RootMinX[2];

  //Setup Initial Conditions
  //rhotot = MU_H*valDens;
  //press = valDens*KB*valTemp;
  //Primitive Variables
  W.Vx = 0.0;
  W.Vy = 0.0;
  W.Vz = 0.0;

  //CLUSTER GENERATION
  //Generate Random Seed
  srand(1246);
  //Generate random phase change (phi)
  int ki, kj, kk;
  double Kmag, localmindld = 1.0, localmaxdld = 0.0;
  deltaLogDens = (double *)calloc(GNx1*GNx2*GNx3,sizeof(double));
  double phix, phiy, phiz;
  double pi2 = 2.0*PI;
#ifdef MPI_PARALLEL
  MPI_Barrier(MPI_COMM_WORLD);
#endif //MPI_PARALLEL

  for(kk=1;kk<3;kk++){
    for(kj=1;kj<3;kj++){
      for(ki=1;ki<3;ki++){
	if(myID_Comm_world == 0){
	  phix = (double)rand()/(double)(RAND_MAX/pi2);
	  phiy = (double)rand()/(double)(RAND_MAX/pi2);
	  phiz = (double)rand()/(double)(RAND_MAX/pi2);
	}
#ifdef MPI_PARALLEL
	MPI_Bcast(&phix,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&phiy,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&phiz,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
#endif //MPI_PARALLEL

	//Find the magnitude of the k-values
	Kmag = sqrt(ki*ki + kj*kj + kk*kk);
	for(k=ks; k<=ke; k++){
	  for(j=js; j<=je;j++){
	    for(i=is; i<=ie;i++){
	      cc_pos(pGrid,i,j,k,&x1,&x2,&x3);
	      deltaLogDens[GNx1*GNx2*(k-ks) + GNx1*(j-js) + (i-is)] += ((1.0/Kmag)*sin(2.0*PI*ki*x1/lx1 + phix)*
										 sin(2.0*PI*kj*x2/lx2 + phiy)*
										 sin(2.0*PI*kk*x3/lx3 + phiz));
	    }
	  }
	}
	//End k-value set. Start new set of k-values
      }
    }
  }

  //Determine Max and Min Cluster Values
  for(k=ks;k<=ke;k++){
    for(j=js;j<=je;j++){
      for(i=is;i<=ie;i++){
	localmindld = FMIN(localmindld, deltaLogDens[GNx1*GNx2*(k-ks) + GNx1*(j-js) + (i-is)]);
	localmaxdld = FMAX(localmaxdld, deltaLogDens[GNx1*GNx2*(k-ks) + GNx1*(j-js) + (i-is)]);
      }
    }
  }
  
#ifdef MPI_PARALLEL
  //Find globalmindld and globalmaxdld using MPI_Allreduce
  MPI_Allreduce(&localmindld,&globalmindld,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
  MPI_Allreduce(&localmaxdld,&globalmaxdld,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  #endif //MPI_PARALLEL

  //Generate Primitive Variables on Domain
  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {
	//Renormalize Cluster Values (values from 0 to 1)
	deltaLogDens[GNx1*GNx2*(k-ks) + GNx1*(j-js) + (i-is)] -= globalmindld;
	deltaLogDens[GNx1*GNx2*(k-ks) + GNx1*(j-js) + (i-is)] /= (globalmaxdld - globalmindld);

	cc_pos(pGrid,i,j,k,&x1,&x2,&x3);
	//Final Primitive Variable Value
	/*rad = 5.0*PCCM; //5 pc radius sphere
	double dmin = 10.0;
	double dmax = 100.0;
	dist = sqrt(x1*x1+x2*x2+x3*x3);
	if(dist <= rad){
	  valDens = dmax;
	  newTemp = valTemp/(dmax/dmin);
	}
	else{
	  valDens = dmin;
	  newTemp = valTemp;
	  }*/

	valDens = pow(10.0,1.0 + deltaLogDens[GNx1*GNx2*(k-ks) + GNx1*(j-js) + (i-is)]);
	rhotot = MU_H*valDens;
	newTemp = 1.0E4/valDens;
	press = (rhotot/MU_H)*KB*newTemp;
	W.d = rhotot;
	W.P = press;
	
	//Store Cons Variables for Prim Variables
	pGrid->U[k][j][i].d  = W.d;
	pGrid->U[k][j][i].M1 = W.d*W.Vx;
	pGrid->U[k][j][i].M2 = W.d*W.Vy;
	pGrid->U[k][j][i].M3 = W.d*W.Vz;
	KE                   = 0.5*W.d*(W.Vx*W.Vx + W.Vy*W.Vy + W.Vz*W.Vz);
	pGrid->U[k][j][i].E  = W.P/(Gamma_1) + KE;

	//S Parameters: H2 = 0, Hp = 1
	pGrid->U[k][j][i].s[0] = valxH2*rhotot;
	pGrid->U[k][j][i].s[1] = valxHp*rhotot;
      }
    }
  }
  
  //Free Cluster Arrays
  free(deltaLogDens);

  //EQM Temperature Interpolation Look-Up Table
  //Dimensions
  dim1 = 30;
  dim2 = 30;
  dim3 = 30;
  //Allocate Input
  powDensarr = (double *)calloc(dim1,sizeof(double));
  powxH2arr = (double *)calloc(dim2,sizeof(double));
  powxHparr = (double *)calloc(dim3,sizeof(double));
  //Allocate Output
  eqmTemparr = (double *)calloc(dim1*dim2*dim3,sizeof(double));
  //Look-Up Table Loop
  for(i=0; i<dim1; i++){
    powDensarr[i] = 0.0 + (4.0 - 0.0)*(1.0*i)/(1.0*dim1-1.0);
    valDens = pow(10.0,powDensarr[i]);
    for(j=0; j<dim2; j++){
      powxH2arr[j] = -8.0 + (8.0)*(1.0*j)/(1.0*dim2-1.0);
      valxH2 = pow(10.0,powxH2arr[j]);
      for(k=0; k<dim3; k++){
	powxHparr[k] = -8.0 + (8.0)*(1.0*k)/(1.0*dim3-1.0);
	valxHp = pow(10.0,powxHparr[k]);
	valTemp = 1000.0;
	eqmTemparr[dim2*dim3*i + dim3*j + k] = GetRoot(valTemp,-1.0);
      }
    }
  }
  
  //Store Min and Max Input Values
  minDens = powDensarr[0];
  maxDens = powDensarr[dim1-1];
  minxH2 = powxH2arr[0];
  maxxH2 = powxH2arr[dim2-1];
  minxHp = powxHparr[0];
  maxxHp = powxHparr[dim3-1];

  //SELF GRAVITY
#ifdef SELF_GRAVITY
  GetMeanRho(pDomain);
#endif //SELF_GRAVITY

  if(myID_Comm_world == 0)
    printf("GravMeanRho = %.3e\n",grav_mean_rho);

}

void problem_write_restart(MeshS *pM, FILE *fp)
{
  return;
}

void problem_read_restart(MeshS *pM, FILE *fp)
{
  return;
}

ConsFun_t get_usr_expr(const char *expr)
{
  return NULL;
}

VOutFun_t get_usr_out_fun(const char *name){
  return NULL;
}

void Userwork_in_loop(MeshS *pM)
{
  //Loop Variables
  int nL,nd;
  DomainS *pDomain;
  GridS *pGrid;
  Prim1DS W;
  Cons1DS U1d;
  int i,is,ie;
  int j,js,je;
  int k,ks,ke;
  Real x1,x2,x3;
  //Parameters from grid cell
  double rhoH2,rhoHp,rhotot,Etot,valxe;
  double Mx,My,Mz,KE;

  //Reset globalMaxH2 for each timestep
  globalMaxH2 = 0.0;
  globalMaxDens = 0.0;

  //Loop over all Domains
  for(nL=0;nL<(pM->NLevels);nL++){
    for(nd=0;nd<(pM->DomainsPerLevel[nL]);nd++){
      if(pM->Domain[nL][nd].Grid != NULL){
	pDomain = &(pM->Domain[nL][nd]);
	pGrid = (pDomain->Grid);
	is = pGrid->is;
	ie = pGrid->ie;
	js = pGrid->js;
	je = pGrid->je;
	ks = pGrid->ks;
	ke = pGrid->ke;
	//Loop over every cell
	for (k=ks; k<=ke; k++) {
	  for (j=js; j<=je; j++) {
	    for (i=is; i<=ie; i++) {
	      cc_pos(pGrid,i,j,k,&x1,&x2,&x3);
	      //Obtain Pos of grid cell
	      xpos = x1; ypos = x2; zpos = x3;
	      //Obtain Mass Densities (H2, Hp)
	      rhoH2  = pGrid->U[k][j][i].s[0];
	      rhoHp  = pGrid->U[k][j][i].s[1];
	      rhotot = pGrid->U[k][j][i].d;
	      //Obtain energy of cell
	      Etot   = pGrid->U[k][j][i].E;
	      //Obtain Momentum densities (x,y,z)
	      Mx = pGrid->U[k][j][i].M1;
	      My = pGrid->U[k][j][i].M2;
	      Mz = pGrid->U[k][j][i].M3;
	      KE = (1.0/(2.0*rhotot))*(Mx*Mx+My*My+Mz*Mz);
	      CheckEnergy(Etot,KE);
	      //Obtain Abundance and Temp Values
	      valxH2 = rhoH2/rhotot;
	      valxHp = rhoHp/rhotot;
	      valDens = rhotot/MU_H;
	      valsV = 1.0E-4;
	      valTemp = (Etot-KE)*(Gamma_1)/(valDens*KB);
	      //Allocate CN and TM Arrays
	      vecnHx = vector(1,NVARCN);
	      vecTemp = vector(1,NVART);
	      //Set Start and End Times
	      tstart = pGrid->time;
	      tend = tstart + pGrid->dt;

	      //Run Chemical Network
	      if(boolCN == 1){
		b4xH2 = valxH2; b4xHp = valxHp;
		vecnHx[1] = GetnH2(valxH2);
		vecnHx[2] = GetnHp(valxHp);
		//Check to see if in eqm
		if(b4xH2 >= 0.99 || b4xHp >= 0.99){
		  double tolx = 1.0E-4; double tolf = 1.0E-4; int ntrial = 100;
		  MNRootInit(vecnHx);
		  valxH2 = GetxH2(vecnHx[1]);
		  valxHp = GetxHp(vecnHx[2]);
		  MultiNewt(ntrial,vecnHx,NVARCN,tolx,tolf);
		  //Find eqm abundances
		  eqmxH2 = GetxH2(vecnHx[1]);
		  eqmxHp = GetxHp(vecnHx[2]);
		  /*if(tstart > 4.19E13){
		    if(xpos == 2.8875E18 && ypos == 4.8125E18 && zpos == 2.8875E18){
		      printf("eqmxH2 = %.5e\teqmxHp = %.5e\n",eqmxH2,eqmxHp);
		      printf("xpos = %.3e\typos = %.3e\t zpos = %.3e\n",xpos,ypos,zpos);
		      printf("b4xH2 = %.5e\tb4xHp = %.5e\n",b4xH2,b4xHp);
		      printf("valDens = %.3e\tvalTemp = %.3e\n",valDens,valTemp);
		    }
		    }*/
		  if(fabs(eqmxH2 - b4xH2)/eqmxH2 < TOL){
		    //System in eqm.Obtain eqm abundance values
		    valxH2 = eqmxH2; valxHp = eqmxHp;
		    a4xH2 = valxH2; a4xHp = valxHp;
		  }
		  else{
		    //Restore previous abundance values
		    valxH2 = b4xH2; valxHp = b4xHp;
		    vecnHx[1] = GetnH2(valxH2);
		    vecnHx[2] = GetnHp(valxHp);
		    //Run Chemical Network
		    RunCN(tstart, tend, vecnHx);
		    valxH2 = GetxH2(vecnHx[1]);
		    valxHp = GetxHp(vecnHx[2]);
		    a4xH2 = valxH2; a4xHp = valxHp;
		  }
		}
		else{
		  RunCN(tstart, tend, vecnHx);
		  valxH2 = GetxH2(vecnHx[1]);
		  valxHp = GetxHp(vecnHx[2]);
		  a4xH2 = valxH2; a4xHp = valxHp;
		}
		//Check if Error in Abundance
		CheckAbundance();
		//Update Cons Variables
		pGrid->U[k][j][i].s[0] += MU_H*(a4xH2 - b4xH2)*valDens;
		pGrid->U[k][j][i].s[1] += MU_H*(a4xHp - b4xHp)*valDens;
	      }
	      //Exclude Chemical Network
	      else{
		b4xH2 = valxH2; b4xHp = valxHp;
		a4xH2 = valxH2; a4xHp = valxHp;
	      }

	      //Run Thermal Model
	      if(boolTM == 1){
		//Find EQM Temp for current conditions
		vecTemp[1] = valTemp;
		b4Temp = valTemp;
		//Root-Interpolation
		//eqmTemp = GetInterpolation(log10(valDens),log10(valxH2),log10(valxHp),eqmTemparr);
		//Root-Iteration
		eqmTemp = GetRoot(valTemp,tstart);
		//Only if valTemp != eqmTemp
		/*if(fabs(eqmTemp - vecTemp[1])/eqmTemp > TOL){
		  RunThermal(tstart,tend,vecTemp);
		  valTemp = vecTemp[1];
		  a4Temp = valTemp;
		  }
		  else{
		  a4Temp = eqmTemp;
		  }*/
		//Run Regardless
		RunThermal(tstart,tend,vecTemp);
		valTemp = vecTemp[1];
		a4Temp = valTemp;
		//Update Cons Variables
		pGrid->U[k][j][i].E    += ((valDens*KB*(a4Temp - b4Temp))/Gamma_1); //cm-3 ergs K-1 K s-1 s
		//Check if Error in Energy
		if((pGrid->U[k][j][i].E) < 0.0){
		  ath_error("UserWork: Energy is LESS than ZERO! \nE = %.1e\nb4Temp = %e\na4Temp = %e\n",
			    pGrid->U[k][j][i].E,b4Temp,a4Temp);
		}
	      }
	      //Exclude Thermal Model
	      else{
		a4Temp = valTemp;
	      }

	      #ifdef MPI_PARALLEL
	      //Find globalmaxH2 using MPI_Allreduce
	      MPI_Allreduce(&valxH2,&globalMaxH2,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
	      MPI_Allreduce(&valDens,&globalMaxDens,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
#endif //MPI_PARALLEL

	      //Free Vectors
	      free_vector(vecnHx,1,NVARCN);
	      free_vector(vecTemp,1,NVART);

	    }
	  }
	}

#ifdef SELF_GRAVITY
	pDomain = &(pM -> Domain[nL][nd]);
	GetMeanRho(pDomain);
#endif //SELF_GRAVITY

      }
    }
  }

  if(myID_Comm_world == 0){
    printf("globalMaxH2 = %.3e\ttime = %.3e\tTemp = %.3e\n",globalMaxH2,tend,valTemp);
    printf("globalMaxDens = %.3e\ttime = %.3e\tTemp = %.3e\n",globalMaxDens,tend,valTemp);
    //printf("time = %.3e\thvalueT = %.3e\n",tend,hvalueT);
  }

}

void Userwork_after_loop(MeshS *pM)
{
  free(powDensarr);
  free(powxH2arr);
  free(powxHparr);
  free(eqmTemparr);
}
 
/*==============================================================================
 * PROBLEM PRIVATE USER FUNCTIONS:
 * ERROR CHECK
 * CheckAbundance()   - Checks if the abundance values are out of range
 * INTERPOLATION
 * GetInterpolation() - Performs 3D Linear Interpolation for eqmTemparr
 * HELPER FUNCTIONS
 * GetAbund()         - Returns Chemical Species Abundances
 * GetnH2()           - Finds value of H2 number density given H2 abundance
 * GetnHp()           - Finds value of H+ number density given H+ abundance
 * GetnH()            - Finds value of H number density given H abundance
 * Getne()            - Finds value of e number density given e abundance
 * GetxH2()           - Finds value of H2 abundance given H2 number density
 * GetxHp()           - Finds value of H+ abundance given H+ number density
 * GetxH()            - Finds value of H abundance given H+ and H2 number densities
 * Getxe()            - Finds value of e abundance given H+ abundance
 * CHEMICAL NETWORK FUNCTIONS
 * RunCN()            - Performs Chemical Network Simulation
 * dnxHdt()           - Finds the rate of change of our Hydrogen species vs time
 * dnH2()             - Finds the rate of change for H2 vs time
 * dnHp()             - Finds the rate of change for H+ vs time
 * k1()               - H2 Formation from grains
 * k2()               - H2 collisional dissociation with H
 * k3()               - H2 collisional dissociation with H2
 * k4()               - H2 Photodissociation 
 * k5()               - H+ Formation (Cosmic Ray Ionization)
 * k6()               - H+ collisional formation with e
 * k7()               - H+ collisional dissociation with e
 * k8()               - H+ dissociation from grains
 * THERMAL MODEL FUNCTIONS
 * RunThermal()       - Performs Thermal Model Simulation
 * dTdt()             - Finds the rate of change of Temperature vs time
 * GetNetCool()       - Finds the net rate of heating/cooling in ergs s-1
 * GetxO()            - Ionization Fraction of Ox with charge exchange
 * Getcerr()          - Charge exchange rates b/w H -> Ox
 * GetLoss()          - Finds the net rate of cooling in ergs s-1
 * GetGain()          - Finds the net rate of heating in ergs s-1
 * spline()           - Cubic spline interpolation method
 * splint()           - Cubic spline interpolation method
 * Getsigv()          - Returns collisional rate in cm3 s-1
 * INTEGRATION FUNCTIONS
 * odeint()           - Calculates new values of y after a defined change in x
 * stiff()            - 4th order Rosenbrock step for integrating stiff ODEs (adaptive step size)
 * RKQS()             - Function implementing 5th order RK step
 * RKCK()             - 5th order RK Cash Karp method (adaptive step size)
 * EQM FUNCTIONS
 * GetRoot()          - Finds the temperature of the gas in thermal eqm using bisectional root finder
 * RootInit()         - Determines root finder boundaries
 *-----------------------------------------------------------------------------*/

//CHECK ERRORS

static void CheckAbundance(){
  const double TOLERANCE = 1.0E-3;
  double valxH = GetxH(GetnH2(valxH2),GetnHp(valxHp));
  double valxHtot = valxH + valxH2 + valxHp;
  if(valxHtot <= 1.0 - TOLERANCE || valxHtot >= 1.0 + TOLERANCE){
    printf("Abundances out of range! valxH2 = %.3e\tvalxHp = %.3e\tvalxH = %.3e\n",valxH2,valxHp,valxH);
    exit(1);
  }
}

static void CheckEnergy(double energyValue, double KEvalue){
  if(energyValue - KEvalue < 0.0){
    printf("UserWork: Energy is LESS than ZERO! \nEtot = %.1e\nEint = %.5e\nKE = %.5e\nprev timestep Temp = %.5e\nb4Temp = %.5e\n",energyValue-KEvalue,energyValue,KEvalue,a4Temp,b4Temp);
    printf("xpos = %.5e\typos = %.5e\tzpos = %.5e\n",xpos,ypos,zpos);
    printf("valDens = %.5e\tvalTemp = %.5e\n",valDens,valTemp);
    exit(1);
  }
}

//INTERPOLATION FUNCTIONS

static double GetInterpolation(double powDens, double powxH2, double powxHp, double *eqmTemparr){

  //Compute Interpolated Value in 3D grid
  //Declare Variables
  int ii,jj,kk;
  int iDens,ixH2,ixHp; //Index value found for interpolation
  double dDens,dxH2,dxHp;
  double fDens,fxH2,fxHp;
  double f1,f2,f3;
  double interpValue; //(dxH2dt,dxHpdt,dTdt)
  //Check that the point is inside our grid
  
  //Dens
  if(powDens < minDens)
    powDens = minDens;
  else if(powDens >= maxDens)
    powDens = maxDens;
  
  //xH2
  if(powxH2 < minxH2)
    powxH2 = minxH2;
  else if(powxH2 > maxxH2){
    printf("Value out of H2 abundance range! powxH2 = %.1e\n",powxH2);
    exit(1);
  }

  //xHp
  if(powxHp > 0.0){
    printf("Value out of H+ abundance range! powxHp = %.3e\n",powxHp);
    exit(1);
  }
  if(powxHp < minxHp)
    powxHp = minxHp;
  else if(powxHp > maxxHp)
    powxHp = maxxHp;
  
  //Find Place in Grid
  //printf("GetInterpolation: Index Values\n");
  dDens = (maxDens - minDens)/(1.0*dim1-1.0);
  iDens = floor((powDens - minDens)/dDens); //Dens
  //printf("iDens = %i\tpowDens = %.1e\n",iDens,powDens);
  dxH2  = (maxxH2 - minxH2)/(1.0*dim2-1.0);
  ixH2  = floor((powxH2 - minxH2)/dxH2); //xH2
  //printf("ixH2 = %i\tpowxH2 = %.1e\n",ixH2,powxH2);
  dxHp  = (maxxHp - minxHp)/(1.0*dim3-1.0);
  ixHp  = floor((powxHp - minxHp)/dxHp); //xHp
  //printf("ixHp = %i\tpowxHp = %.1e\n",ixHp,powxHp);
  //Do a Simple Multidimensional Linear Interpolation (3D)
  //Create helper variables for each index
  int un = 1.0;
  //Fraction away from next array value
  fDens = (powDens - (minDens + dDens*iDens))/dDens;
  fxH2  = (powxH2 - (minxH2 + dxH2*ixH2))/dxH2;
  fxHp  = (powxHp - (minxHp + dxHp*ixHp))/dxHp;
  //To make syntax easier
  f1 = fDens;
  f2 = fxH2;
  f3 = fxHp;
  //Perform 3D Interpolation
  //Declare all 8 points in 3D surface (eqmTemparr)
  double y[2][2][2];
  for(ii=0;ii<2;ii++){
    for(jj=0;jj<2;jj++){
      for(kk=0;kk<2;kk++){

	if(iDens == dim1-1 && ixH2 == dim2-1 && ixHp == dim3-1)
	  y[ii][jj][kk] = ARReqmTemp(iDens,ixH2,ixHp);
	else if(iDens == dim1-1 && ixH2 == dim2-1)
	  y[ii][jj][kk] = ARReqmTemp(iDens,ixH2,ixHp+kk);
	else if(iDens == dim1-1 && ixHp == dim3-1)
	  y[ii][jj][kk] = ARReqmTemp(iDens-1,ixH2+jj,ixHp);
	else if(iDens == dim1-1)
	  y[ii][jj][kk] = ARReqmTemp(iDens,ixH2+jj,ixHp+kk);
	else if(ixH2 == dim2-1)
	  y[ii][jj][kk] = ARReqmTemp(iDens+ii,ixH2,ixHp+kk);
	else if(ixHp == dim3-1)
	  y[ii][jj][kk] = ARReqmTemp(iDens+ii,ixH2+jj,ixHp);
	else
	  y[ii][jj][kk] = ARReqmTemp(iDens+ii,ixH2+jj,ixHp+kk);

      }
    }
  }

  //Perform Interpolation
  interpValue = (       (un-f1)*(un-f2)*(un-f3)*y[0][0][0] +
                        (un-f1)*(un-f2)*f3     *y[0][0][1] +
			(un-f1)*f2     *(un-f3)*y[0][1][0] +
			(un-f1)*f2     *f3     *y[0][1][1] +
			f1     *(un-f2)*(un-f3)*y[1][0][0] + 
			f1     *(un-f2)*f3     *y[1][0][1] +
			f1     *f2     *(un-f3)*y[1][1][0] +
			f1     *f2     *f3     *y[1][1][1] );

  //fprintf(fileRootInterp,"iDens = %i\tixH2 = %i\tixHp = %i\tinterpValue = %e\n",iDens,ixH2,ixHp,interpValue);

  return interpValue;
}

//ITERATION FUNCTIONS

static double GetRoot(double Tmid, double time)
/*Thermal Bisectional Root Finder
  1) Determine Steady State Ionization fractions for each temp and dens
  2) Bracket off root using previous root as starting point
  3) Use above temps in bisectional root finder to find Tmid
*/
{
  void RootInit(double Tmid, double Tbounds[]);
  //Declare Variables
  double Thi, Tlo;
  double *Tbounds = vector(1,2);
  double fhi, fmid, flo;
  double EPSILON = 1.0E-5;
  //Find Root Bounds
  fmid = GetNetCool(Tmid);
  RootInit(Tmid, Tbounds);
  Tlo = Tbounds[1];
  Thi = Tbounds[2];
  //Find net heating/cooling for bounds
  fhi = GetNetCool(Thi);
  flo = GetNetCool(Tlo);
  Tmid = Tlo + (Thi - Tlo)/2.0;
  //Perform bisectional root find
  while(fabs(Thi - Tlo)/Tmid > EPSILON){
    fmid = GetNetCool(Tmid);
    if(fmid*flo < 0.0){
      Thi = Tmid;
      fhi = fmid;
    }
    else{
      Tlo = Tmid;
      flo = fmid;
    }
    Tmid = Tlo + (Thi - Tlo)/2.0;
  }
  //printf("a4: Tlo = %.3e\t Thi = %.3e\t Tmid = %.3e\t time = %e\n",Tlo,Thi,Tmid,time);
  free_vector(Tbounds,1,2);
  return Tmid;
}

void RootInit(double Tmid, double Tbounds[])
//Root Finder Boundaries
{
  //Declare Variables
  double netcool, i, j;
  double fhi, flo;

  //Find netcool at current temp 
  netcool = GetNetCool(Tmid);
  if(netcool <= 0.0){ //cooling > heating! T decreases
    Tbounds[2] = Tmid*1.1;
    fhi = GetNetCool(Tbounds[2]);
    flo = fhi;
    Tbounds[1] = Tmid;
    i = 1.0;
    while(fhi*flo > 0.0 && Tbounds[1] >= 3.0E0){
      Tbounds[1] = Tmid*pow(0.9,i);
      flo = GetNetCool(Tbounds[1]);
      i += 1.0;
    }
  }else{ //heating > cooling! T increases
    Tbounds[1] = 0.9*Tmid;
    flo = GetNetCool(Tbounds[1]);
    fhi = flo;
    Tbounds[2] = Tmid;
    j = 1.0;
    while(fhi*flo > 0.0 && Tbounds[2] <= 1.0E4){
      Tbounds[2] = Tmid*pow(1.1,j);
      fhi = GetNetCool(Tbounds[2]);
      j += 1.0;
    }
  }
}

//HELPER FUNCTIONS
 
static void GetAbund(double abund[])
{
  abund[1] = 1.0E-1;
  abund[2] = 3.2E-4;
  abund[3] = 1.4E-4;
  abund[4] = 1.7E-6;
  //Total Hydrogen Abundance
  abund[0] = 1.0 - abund[1] - abund[2] - abund[3] - abund[4];
}

static double GetnH2(double xH2) 
{
  double nH2;
  nH2 = valDens*xH2*abund[0]/2.;
  return nH2;
}

static double GetnHp(double xHp)
{
  double nHp;
  nHp = valDens*xHp*abund[0];
  return nHp;
}

static double GetnH(double nH2, double nHp)
{
  double nH;
  nH = valDens*abund[0]*GetxH(nH2,nHp);
  return nH;
}

static double Getne(double nHp)
{
  double ne;
  ne = Getxe(nHp)*valDens;
  return ne;
}

static double GetxH2(double nH2)
{
  double xH2;
  xH2 = 2.*nH2/(valDens*abund[0]);
  return xH2;
}

static double GetxHp(double nHp)
{
  double xHp;
  xHp = nHp/(valDens*abund[0]);
  return xHp;
}

static double GetxH(double nH2, double nHp)
{
  double xH;
  xH = 1.0 - GetxH2(nH2) - GetxHp(nHp);
  return xH;
}

static double Getxe(double nHp)
{
  double xe;
  xe = abund[0]*GetxHp(nHp) + abund[3] + abund[4];
  return xe;
}

//CHEMICAL NETWORK FUNCTIONS

static void RunCN(double tstart, double tend, double vecnHx[])
{ 
  xp = vector(1,KMAX);
  yp = matrix(1,NVARCN,1,KMAX);
  dxsav = tend/(KMAX*1.2);
  int nok = 0, nbad = 0;
  odeint(vecnHx, NVARCN, tstart, tend, EPS, hvalueCN, HMIN, &nok, &nbad, 1, &dnHxdt, &stiff);
  valxH2 = GetxH2(vecnHx[1]);
  valxHp = GetxHp(vecnHx[2]);

  //Free Vectors
  free_vector(xp,1,KMAX);
  free_matrix(yp,1,NVARCN,1,KMAX);
} 

static void dnHxdt(double valt, int nvar, double arrnHx[], double dnHxdt[])
{
  valxH2 = GetxH2(arrnHx[1]);
  //Error Check
  /*if(valxH2 < 0.0){
    printf("dnHxdt: valxH2 = %.3e. Floor value!\n",valxH2);
    valxH2 = 1.0E-8;
    arrnHx[1] = GetnH2(valxH2);
    }*/
  valxHp = GetxHp(arrnHx[2]);
  //Error Check
  /*if(valxHp < 0.0){
    printf("dnHxdt: valxHp = %.3e. Floor value!",valxHp);
    valxHp = 1.0E-8;
    arrnHx[2] = GetnHp(valxHp);
    }*/
  dnHxdt[1] = dnH2(arrnHx);
  dnHxdt[2] = dnHp(arrnHx);
  int i;
  for(i=1;i<=NVARCN;i++){
    if(arrnHx[i] != arrnHx[i]){
      printf("arrnHx[%i] is NaN. Exit Now!\n",i);
      printf("xpos = %.5e\typos = %.5e\tzpos = %.5e\n",xpos,ypos,zpos);
      exit(1);
    }
    if(dnHxdt[i] != dnHxdt[i]){
      printf("dnHxdt[%i] is NaN. Exit Now!\n",i);
      printf("k1 = %.5e\tk2 = %.5e\tk3 = %.5e\tk4 = %.5e\n",
	     k1(valTemp,GRAIN_TEMP),k2(valTemp,GetnH2(valxH2),GetnHp(valxHp)),
	     k3(valTemp,GetnH2(valxH2),GetnHp(valxHp)),k4ss());
      printf("xpos = %.5e\typos = %.5e\tzpos = %.5e\n",xpos,ypos,zpos);
      printf("valDens = %.5e\tvalTemp = %.5e\n",valDens,valTemp);
      printf("xH2 = %.5e\txHp = %.5e\txH = %.5e\n",valxH2,valxHp,GetxH(vecnHx[1],vecnHx[2]));
      exit(1);
    }
  }
}

static double dnH2(double nHx[])
/*Rate of H2 formation/dissociation terms*/
{
  double nH2, nHp, nH, dnH2;
  nH2 = nHx[1];
  nHp = nHx[2];
  //Error Check
  if(nHp <= 0.0 || nH2 <= 0.0){
    printf("dnH2: nHx values impossible!\tnHp = %.3e\tnH2 = %.3e\tNTOT = %.3e\n",nHx[2],nHx[1],valDens);
    printf("xpos = %.3e\typos = %.3e\tzpos = %.3e\n",xpos,ypos,zpos);
    exit(1);
  }
  nH = GetnH(nH2,nHp);
  if(nH < 0.0){
    double xH2 = GetxH2(nH2);
    double xHp = GetxHp(nHp);
    double xH = GetxH(nH2,nHp);
    printf("xH2 = %.3e\txH+ = %.3e\txH = %.3e\n",xH2,xHp,xH);
    printf("eqmxH2 = %.5e\teqmxHp = %.5e\n",eqmxH2,eqmxHp);
    double xTot = xH + xH2 + xHp;
    printf("xTot = %.5e\tvalDens = %.3e\tvalTemp = %.3e\n",xTot,valDens,valTemp);
    printf("Exiting Now!\txpos = %.3e\typos= %.3e\tzpos = %.3e\n",xpos,ypos,zpos);
    exit(1);
  }
  //dnH2 = (k1(valTemp,GRAIN_TEMP)*nH*nH - k2(valTemp,nH2,nHp)*nH2*nH - 
  //k3(valTemp,nH2,nHp)*nH2*nH2 - k4ss()*nH2);
  dnH2 = k1(valTemp,GRAIN_TEMP)*nH*nH - k4ss()*nH2;
  return dnH2;
}

static double dnHp(double nHx[])
{
  double nH2, nHp, nH, ne, dnHp;
  nH2 = nHx[1];
  nHp = nHx[2];
  //Error Check
  if(nHp <= 0.0 || nH2 <= 0.0){
    printf("dnHp: nHx values impossible!\tnHp = %.3e\tnH2 = %.3e\tNTOT = %.3e\n",nHx[2],nHx[1],valDens);
    exit(1);
  }
  nH = GetnH(nH2,nHp);
  ne = Getne(nHp);
  dnHp = k5()*nH + k6(valTemp)*nH*ne - k7(valTemp)*nHp*ne - k8(valTemp,ne)*nHp*ne;
  return dnHp;
}

static double k1(double temp, double tempgrain)
/*H2 Formation from Grain Catalysis HM 1979*/
{
  double fa, t2, tgrain2, k1;
  fa = 1.0/(1.0+1.0E4*exp(-600.0/tempgrain));
  t2 = 1.0E-2*temp;
  tgrain2 = 1.0E-2*tempgrain;
  k1 = (3.0E-17*sqrt(t2)*fa)/(1.0+0.4*sqrt(t2+tgrain2)+0.2*t2+0.08*t2*t2);
  return k1;
}

static double k2(double temp, double nH2, double nHp)
/*H2 Collisional Dissociation w H MacLow and Shull 1986(low) Lepp and Shull1983(high)*/
{
  double kH, kL, logTemp, ncrH, ncrH2, ndncr, k2;
  kH = 3.52E-9*exp(-4.39E4/temp);
  kL = 6.67E-12*sqrt(temp)*exp(-1.0*(1.0+63590.0/temp));
  logTemp = log10(1.0E-4*temp);
  ncrH = pow(10.0,3.0-0.416*logTemp-0.327*logTemp*logTemp);
  ncrH2 = pow(10.0,4.845-1.3*logTemp+1.62*logTemp*logTemp);
  ndncr = valDens*((GetxH(nH2,nHp)/ncrH)+(GetxH2(nH2)/ncrH2));
  k2 = pow(kH,(ndncr/(1.0+ndncr)))*pow(kL,1.0/(1.0+ndncr));
  return k2;
}

static double k3(double temp, double nH2, double nHp)
/*H2 collisional dissociation w H2 Martin 1998(low) Shapiro and Kang 1987(high)*/
{
  double kH, kL, logTemp, ncrH, ncrH2, ndncr, k3;
  kH = 1.3E-9*exp(-5.33E4/temp);
  kL = 1.18E-10*exp(-6.95E4/temp);
  logTemp = log10(1.0E-4*temp);
  ncrH = pow(10.0,3.0-0.416*logTemp-0.327*logTemp*logTemp);
  ncrH2 = pow(10.0,4.845-1.3*logTemp+1.62*logTemp*logTemp);
  ndncr = valDens*((GetxH(nH2,nHp)/ncrH)+(GetxH2(nH2)/ncrH2));
  k3 = pow(kH,(ndncr/(1.0+ndncr)))*pow(kL,1.0/(1.0+ndncr));
  return k3;
}

static double k4()
/*H2 Photodissociation Rate (Based on Strength of Radiation Field)*/
{
  double kph0, k4;
  kph0 = CHI*3.3E-11; //define CHI
  k4 = valsV*kph0;
  return k4;
}

static double k4ss()
//H2 Photodissociation rate (Local self shielding approximation) GM2007 2.2.1
{
  //Declare variables
  double kph0, delX, cdHtot, cdH2, x, fshield, k4;
  kph0 = CHI*3.3E-11; //s-1
  delX = 2.0*x1max/(1.0*Nx1);
  cdHtot = (delX/2.0)*valDens*abund[0];
  cdH2 = cdHtot*valxH2;
  x = cdH2/(5.0E14);
  fshield = ((0.965/((1.0+x)*(1.0+x))) + (0.035/sqrt(1.0+x))*exp(-8.5E-4*sqrt(1.0+x)));
  k4 = fshield*exp(-2.0E-21*cdHtot)*kph0;
  return k4;
}

static double k5()
/*H+ Formation due to Cosmic ray ionization*/
{
  double k5;
  k5 = ZETA;
  return ZETA;
}

static double k6(double temp)
/*H+ collisional formation w e Draine eqn 13.11*/
{
  double k6;
  k6 = 5.466E-9*1.07*sqrt(1.0E-4*temp)*exp(-13.6/(8.6173E-5*temp));
  return k6;
}

static double k7(double temp)
/*H+ collisional dissociation Draine case B eqn 14.6*/
{
  double t4, k7;
  t4 = 1.0E-4*temp;
  k7 = 2.54E-13*pow(t4,-0.8163-0.0208*log(t4));
  return k7;
}

static double k8(double temp, double ne)
/*H+ dissociation from grains*/
{
  double G0, psi, k8;
  G0 = 1.13;
  psi = (G0*sqrt(temp))/ne;
  k8 = (1.0E-14*12.25)/(1.0+8.074E-6*pow(psi,1.378)*(1.0+5.087E2*pow(temp,1.586E-2)*pow(psi,-0.4723-1.102E-5*log(temp))));
  return k8;
}

//MULTINEWT FUNCTION

static void MNRootInit(double nHxmid[])
//Root Finder Boundaries
{
  //Declare Variables
  double netcool, i, j, k, m;
  double dnH2hi, dnH2lo;
  double dnHphi, dnHplo;
  double maxnH2, maxnHp;
  double **nHxbounds = matrix(1,2,1,2);
  double *dnHxmid = vector(1,2);
  double *usednHx = vector(1,2);

  //Find netcool at current temp 
  double valt = 0.0;
  dnHxdt(valt, NVARCN, nHxmid, dnHxmid);
  //printf("Finding Case: dnH2 = %.5e\tdnHp = %.5e\n",dnHxmid[1],dnHxmid[2]);
  if(dnHxmid[1] <= 0.0 && dnHxmid[2] <= 0.0){ //Destruction > Formation! nHx decreases
    printf("Destruction > Formation! nHx decreases\n");
    nHxbounds[1][2] = nHxmid[1]*1.01;
    nHxbounds[2][2] = nHxmid[2]*1.01;
    usednHx[1] = nHxbounds[1][2]; usednHx[2] = nHxbounds[2][2];
    dnHxdt(valt, NVARCN, usednHx, dnHxmid);
    dnH2hi = dnHxmid[1]; dnHphi = dnHxmid[2];
    dnH2lo = dnH2hi; dnHplo = dnHphi;
    nHxbounds[1][1] = nHxmid[1];
    nHxbounds[2][1] = nHxmid[2];
    i = 1.0;
    //printf("dnH2hi = %.3e\tdnHphi = %.3e\n",dnH2hi,dnHphi);
    while(dnH2hi*dnH2lo > 0.0 || dnHphi*dnHplo > 0.0){
      //printf("nH2 = %.3e\tnH+ = %.3e\n",usednHx[1],usednHx[2]);
      if(dnH2hi*dnH2lo > 0.0) nHxbounds[1][1] = nHxmid[1]*pow(0.99,i);
      if(dnHphi*dnHplo > 0.0) nHxbounds[2][1] = nHxmid[2]*pow(0.99,i);
      usednHx[1] = nHxbounds[1][1]; usednHx[2] = nHxbounds[2][1];
      dnHxdt(valt, NVARCN, usednHx, dnHxmid);
      dnH2lo = dnHxmid[1]; dnHplo = dnHxmid[2];
      //printf("dnH2lo = %.7e\tdnHplo = %.3e\n",dnH2lo,dnHplo);
      i += 1.0;
    }
    nHxmid[1] = nHxbounds[1][1];
    nHxmid[2] = nHxbounds[2][1];
  }
  else if(dnHxmid[1] > 0.0 && dnHxmid[2] > 0.0){ //Formation > Destruction! nHx increases
    //printf("Formation > Destruction! nHx increases\n");
    nHxbounds[1][1] = nHxmid[1]*0.99;
    nHxbounds[2][1] = nHxmid[2]*0.99;
    usednHx[1] = nHxbounds[1][1]; usednHx[2] = nHxbounds[2][1];
    dnHxdt(valt, NVARCN, usednHx, dnHxmid);
    dnH2lo = dnHxmid[1]; dnHplo = dnHxmid[2];
    dnH2hi = dnH2lo; dnHphi = dnHplo;
    nHxbounds[1][2] = nHxmid[1];
    nHxbounds[2][2] = nHxmid[2];
    j = 1.0;
    maxnH2 = GetnH2(1.0);
    maxnHp = GetnHp(1.0);
    //printf("dnH2lo = %.3e\tdnHplo = %.3e\n",dnH2lo,dnHplo);
    while((dnH2hi*dnH2lo > 0.0 || dnHphi*dnHplo > 0.0) && (nHxmid[1]*pow(1.01,j) < maxnH2 && nHxmid[2]*pow(1.01,j) < maxnHp)){
      //printf("nH2 = %.3e\tnH+ = %.3e\n",usednHx[1],usednHx[2]);
      if(dnH2hi*dnH2lo > 0.0) nHxbounds[1][2] = nHxmid[1]*pow(1.01,j);
      if(dnHphi*dnHplo > 0.0) nHxbounds[2][2] = nHxmid[2]*pow(1.01,j);
      usednHx[1] = nHxbounds[1][2]; usednHx[2] = nHxbounds[2][2];
      dnHxdt(valt, NVARCN, usednHx, dnHxmid);
      dnH2hi = dnHxmid[1]; dnHphi = dnHxmid[2];
      //printf("dnH2hi = %.7e\tdnHphi = %.3e\n",dnH2hi,dnHphi);
      j += 1.0;
    }
    nHxmid[1] = nHxbounds[1][2];
    nHxmid[2] = nHxbounds[2][2];
  }
  else if(dnHxmid[1] > 0.0 && dnHxmid[2] <= 0.0){ //Formation > Destruction! nH2 increases. Destruction> Formation! nHp decreases
    //printf("Formation > Destruction! nH2 increases. Destruction > Formation! nH+ decreases\n");
    nHxbounds[1][1] = nHxmid[1]*0.99;
    nHxbounds[2][2] = nHxmid[2]*1.01;
    usednHx[1] = nHxbounds[1][1]; usednHx[2] = nHxbounds[2][2];
    dnHxdt(valt, NVARCN, usednHx, dnHxmid);
    dnH2lo = dnHxmid[1]; dnHphi = dnHxmid[2];
    dnH2hi = dnH2lo; dnHplo = dnHphi;
    nHxbounds[1][2] = nHxmid[1];
    nHxbounds[2][1] = nHxmid[2];
    k = 1.0;
    //printf("dnH2lo = %.3e\tdnHphi = %.3e\n",dnH2lo,dnHphi);
    maxnH2 = GetnH2(1.0);
    //printf("maxnH2 = %.5e\tabund[0] = %.5e\n",maxnH2,abund[0]);
    while((dnH2hi*dnH2lo > 0.0 || dnHphi*dnHplo > 0.0) && nHxmid[1]*pow(1.01,k) < maxnH2){
      //printf("nH2 = %.3e\tnH+ = %.3e\n",usednHx[1],usednHx[2]);
      if(dnH2hi*dnH2lo > 0.0) nHxbounds[1][2] = nHxmid[1]*pow(1.01,k);
      if(dnHphi*dnHplo > 0.0) nHxbounds[2][1] = nHxmid[2]*pow(0.99,k);
      usednHx[1] = nHxbounds[1][2]; usednHx[2] = nHxbounds[2][1];
      dnHxdt(valt, NVARCN, usednHx, dnHxmid);
      dnH2hi = dnHxmid[1]; dnHplo = dnHxmid[2];
      //printf("dnH2hi = %.3e\tdnHplo = %.3e\n",dnH2hi,dnHplo);
      k += 1.0;
    }
    nHxmid[1] = nHxbounds[1][2];
    nHxmid[2] = nHxbounds[2][1];
    //printf("Ended root bounding\n");
  }
  else if(dnHxmid[1] <= 0.0 && dnHxmid[2] > 0.0){ //Destruction > Formation! nH2 decreases. Formation> Destruction! nHp increases
    //printf("Destruction > Formation! nH2 decreases. Formation > Destruction! nH+ increases\n");
    nHxbounds[1][2] = nHxmid[1]*1.01;
    nHxbounds[2][1] = nHxmid[2]*0.99;
    usednHx[1] = nHxbounds[1][2]; usednHx[2] = nHxbounds[2][1];
    dnHxdt(valt, NVARCN, usednHx, dnHxmid);
    dnH2hi = dnHxmid[1]; dnHplo = dnHxmid[2];
    dnH2lo = dnH2hi; dnHphi = dnHplo;

    nHxbounds[1][1] = nHxmid[1];
    nHxbounds[2][2] = nHxmid[2];
    m = 1.0;
    maxnHp = GetnHp(1.0);
    //printf("dnH2hi = %.3e\tdnHplo = %.3e\n",dnH2hi,dnHplo);
    while((dnH2hi*dnH2lo > 0.0 || dnHphi*dnHplo > 0.0) && nHxmid[2]*pow(1.01,m) < maxnHp){
      //printf("nH2 = %.3e\tnH+ = %.3e\n",usednHx[1],usednHx[2]);
      if(dnH2hi*dnH2lo > 0.0) nHxbounds[1][1] = nHxmid[1]*pow(0.99,m);
      if(dnHphi*dnHplo > 0.0) nHxbounds[2][2] = nHxmid[2]*pow(1.01,m);
      usednHx[1] = nHxbounds[1][1]; usednHx[2] = nHxbounds[2][2];
      dnHxdt(valt, NVARCN, usednHx, dnHxmid);
      dnH2lo = dnHxmid[1]; dnHphi = dnHxmid[2];
      //printf("dnH2lo = %.3e\tdnHphi = %.3e\n",dnH2lo,dnHphi);
      m += 1.0;
    }
    nHxmid[1] = nHxbounds[1][1];
    nHxmid[2] = nHxbounds[2][2];

  }
  //printf("End Init: nH2 = %.3e\tnH+ = %.3e\n",nHxmid[1],nHxmid[2]);
  free_vector(dnHxmid,1,2);
  free_vector(usednHx,1,2);
  free_matrix(nHxbounds,1,2,1,2);
}


#define FREERETURN {free_matrix(fjac,1,n,1,n);free_vector(fvec,1,n);\
    free_vector(p,1,n);free_ivector(indx,1,n);return;}

static void MultiNewt(int ntrial, double x[], int n, double tolx, double tolf)
/*
Given an initial guess x[1..n] for a root in n dimensions, take ntrial 
Newton-Raphson steps to improve the root. Stop if the root converges
ine either summed absolute variable increments tolx or summed
absolute function values tolf
*/
{
  void usrfun(double *x, int n, double *fvec, double **fjac);
  void lubksb(double **a, int n, int *indx, double b[]);
  void ludcmp(double **a, int n, int *indx, double *d);

  int k,i,*indx;
  double errx,errf,d,*fvec,**fjac,*p;
  double pastnH2, pastnHp;
  double pastdnH2, pastdnHp;

  indx=ivector(1,n);
  p = vector(1,n);
  fvec = vector(1,n);
  fvec[1] = 0.0; fvec[2] = 0.0;
  fjac = matrix(1,n,1,n);

  for(k=1;k<=ntrial;k++){
    pastnH2 = x[1]; pastnHp = x[2];
    pastdnH2 = fvec[1]; pastdnHp = fvec[2];
    usrfun(x,n,fvec,fjac); //User supplies function values at x in fvec and jac in fjac
    //printf("fvec[1] = %.3e\tfvec[2] = %.3e\n",fvec[1],fvec[2]);
    //errf = 0.0;
    //for(i=1;i<=n;i++) errf += fabs(fvec[i]); //Check function convergence
    //if(errf <= tolf) FREERETURN;
    if(fabs(fvec[1] - pastdnH2)/fabs(pastdnH2) < tolf && 
       fabs(fvec[2] - pastdnHp)/fabs(pastdnHp) < tolf){
      //printf("dnH2% = %.5e\tdnH+% = %.5e\n",fabs(fvec[1] - pastdnH2)/fabs(pastdnH2), 
      //fabs(fvec[2] - pastdnHp)/fabs(pastdnHp));
      FREERETURN;
    }
    for(i=1;i<=n;i++) p[i] = -fvec[i]; //Right hand side of linear equation
    ludcmp(fjac,n,indx,&d); //Solve linear equations using LU decomposition
    lubksb(fjac,n,indx,p);
    errx = 0.0;
    for(i=1;i<=n;i++){
      errx += fabs(p[i]);
      x[i] += p[i];
    }
    //if(errx <= tolx) FREERETURN;
    if(fabs(GetxH2(pastnH2) - GetxH2(x[1]))/GetxH2(pastnH2) < tolx ||
       fabs(GetxHp(pastnHp) - GetxHp(x[2]))/GetxHp(pastnHp) < tolf) FREERETURN;
    //printf("Trial # = %i\txH2 = %.3e\txHp = %.3e\n",k,GetxH2(x[1]),GetxHp(x[2]));
  }
  FREERETURN;
}

void usrfun(double *x, int n, double *fvec, double **fjac)
{
  void jacobn(double x, double y[], double dfdx[], double **dfdy, int n);
  double time = 0.0;
  //Find dnHxdt values
  dnHxdt(time, n, x, fvec);
  //Find Jacobian values
  double *dfdx = vector(1,n);
  jacobn(time, x,dfdx,fjac,n);
}

//THERMAL MODEL FUNCTIONS

static void RunThermal(double tstart, double tend, double vecTemp[])
{
  xp = vector(1,KMAX);
  yp = matrix(1,NVART,1,KMAX);
  dxsav = tend/(KMAX*1.2);
  int nok = 0, nbad = 0;
  odeint(vecTemp, NVART, tstart, tend, EPS, hvalueT, HMIN, &nok, &nbad, 0, &dTdt, &RKQS);

  //Free Vectors
  free_vector(xp,1,KMAX);
  free_matrix(yp,1,NVART,1,KMAX);
}

static void dTdt(double valt, int nvar, double arrTemp[], double dTdt[])
{
  double coolrate, xe;
  coolrate = valDens*GetNetCool(arrTemp[1]); //ergs cm-3 s-1
  xe = Getxe(GetnHp(valxHp));
  //Gamma = (5.5 + 5.0*xe - 1.5*valxH2)/(3.3 + 3.0*xe - 0.5*valxH2);
  dTdt[1] = ((Gamma_1)/(KB*valDens))*coolrate;
}

static double GetNetCool(double temp)
/*Net heating/cooling in ergs s-1*/
{
  double xe, xO, netloss, netgain, Lambda;
  double GetxO();
  xe = Getxe(GetnHp(valxHp));
  xO = GetxO(); //Charge Exchange
  netloss = GetLoss(temp,xe,xO);
  netgain = GetGain(temp,xe);
  Lambda = netgain - netloss;
  return Lambda;
}

double GetxO()
{
  //Calculates the ionization fraction of oxygen at a steady state based off of charge exchange
  double rate[4],xO;
  void Getcerr(double rate[]); 
  Getcerr(rate);
  xO = (rate[3]*valxHp)/((rate[0] + rate[1] + rate[2])*GetxH(GetnH2(valxH2),GetnHp(valxHp)) + rate[3]*valxHp);
  return xO;
}

void Getcerr(double rate[])
//Reaction rates for charge exchange O -> H+
{
  double T4 = valTemp*1.0E-4;
  rate[0] = (1.14E-9)*pow(T4,0.4+0.018*log(T4));
  rate[1] = (3.44E-10)*pow(T4,0.451+0.036*log(T4));
  rate[2] = (5.33E-10)*pow(T4,0.384+0.024*log(T4))*exp(-97.0/valTemp);
  rate[3] = 1.6*rate[0]*exp(-229.0/valTemp);
}

static double GetLoss(double T, double xe, double xO)
/*Net cooling terms in ergs s-1*/
{
  double losstot, lossC, lossO, lossSi, lossH, lossRec;
  double GetLossCH(double T);
  double GetLossCE(double T, double xe);
  double GetLossCH2(double T);
  double GetLossOxE(double T, double xe, double xO);
  double GetLossOxHp(double T, double xO);
  double GetLossOxH(double T, double xO);
  double GetLossSiE(double T, double xe);
  double GetLossSiH(double T);
  double GetLossHE(double T, double xe);
  double GetLossRec(double T, double xe);
  lossC = GetLossCH(T) + GetLossCE(T,xe) + GetLossCH2(T);
  lossO = GetLossOxE(T,xe,xO) + GetLossOxHp(T,xO) + GetLossOxH(T, xO);
  lossSi = GetLossSiH(T) + GetLossSiE(T,xe);
  lossH = GetLossHE(T,xe);
  lossRec = GetLossRec(T,xe);
  losstot = lossH + lossC + lossO + lossSi + lossRec;
  return losstot;
}

/*=============================================================================
 * GETLOSS FUNCTION HELPER FUNCTIONS
 * GetLossCH()        - Carbon cooling due to collision w H
 * GetLossCE()        - Carbon cooling due to collision w e
 * GetLossCH2()       - Carbon cooling due to collision w H2
 * GetLossOxE()       - Oxygen cooling due to collision w e
 * GetLossOxHp()      - Oxygen cooling due to collision w Hp
 * GetLossOxH()       - Oxygen cooling due to collision w H
 * GetLossSiH()       - Silicon cooling due to collision w H
 * GetLossSiE()       - Silicon cooling due to collision w e
 * GetLossHE()        - Hydrogen cooling due to collision w e
 * GetLossRec()       - Cooling due to radative recombination
 * GetCollHM()        - Collisional rates calculator from HM1989
 *----------------------------------------------------------------------------*/

double GetLossCH(double T)
/*Carbon cooling due to collision w H*/
{
  //Declare variables
  double T2, hcl, exponent;
  double RCH, *powTk, *Tk, *Rk, *Rk2;
  double lossCH;
  T2 = T*1.0E-2;
  hcl = HPLANCK*CSPEED/(157.7E-4);
  exponent = hcl/KB;
  
  if(T <= 2000.0){
    //HM 1989 C -> H 
    RCH = (8.0E-10)*pow(T2,0.07);
  }else if(T > 2000.0){
    //Keenan 1986 C -> H Table 2: T > 2000K
    //Temp Table
    powTk = vector(1,12);
    powTk[1] = 1.0; powTk[2] = 2.0; powTk[3] = 2.5; powTk[4] = 3.0; powTk[5] = 3.25;
    powTk[6] = 3.5; powTk[7] = 3.75; powTk[8] = 4.0; powTk[9] = 4.25; powTk[10] = 4.5;
    powTk[11] = 4.75; powTk[12] = 5.0;
    Tk = vector(1,12);
    int i;
    for(i = 1; i<=12; i++){
      Tk[i] = pow(10.0,powTk[i]);
    }
    //excitation rate cm3 s-1 (*E9)
    Rk = vector(1,12);
    Rk[1] = 1.49E-4; Rk[2] = 6.48E-1; Rk[3] = 1.27; Rk[4] = 1.61; Rk[5] = 1.85;
    Rk[6] = 2.21; Rk[7] = 2.72; Rk[8] = 3.41; Rk[9] = 4.35; Rk[10] = 5.61;
    Rk[11] = 7.29; Rk[12] = 9.49;
    Rk2 = vector(1,12);
    //Interpolation
    spline(Tk, Rk, 12, ((Rk[2] - Rk[1])/(Tk[2] - Tk[1])), ((Rk[12] - Rk[11])/(Tk[12] - Tk[11])), Rk2);
    splint(Tk, Rk, Rk2, 12, T, &RCH);
    RCH *= 1.0E-9;
    //Free vectors
    free_vector(powTk,1,12);
    free_vector(Tk,1,12);
    free_vector(Rk,1,12);
    free_vector(Rk2,1,12);
  }
  
  lossCH = valDens*abund[aC]*abund[aH]*GetxH(GetnH2(valxH2),GetnHp(valxHp))*hcl*RCH*exp(-exponent/T); //ergs s-1
  return lossCH;
}

double GetLossCE(double T, double xe)
/*Carbon cooling due to collision w e*/
{
  //Declare Variables
  double hcl, exponent;
  double *powTw, *Tw, *csw, *csw2,csCE;
  double Lw, lossCE;
  //Wilson Bell 2002 CII -> e- pg 1030
  hcl = HPLANCK*CSPEED/(157.7E-4);
  exponent = hcl/KB;
  //Temp Table
  powTw = vector(1,9);
  powTw[1] = 3.0; powTw[2] = 3.25; powTw[3] = 3.5; powTw[4] = 3.75; powTw[5] = 4.0;
  powTw[6] = 4.25; powTw[7] = 4.5; powTw[8] = 4.75; powTw[9] = 5.0;
  Tw = vector(1,9);
  int i;
  for(i = 1; i <= 9; i++){
    Tw[i] = pow(10.0,powTw[i]);
  }
  //Collision Strength 
  csw = vector(1,9);
  csw[1] = 1.78; csw[2] = 1.80; csw[3] = 1.82; csw[4] = 1.95; csw[5] = 2.24;
  csw[6] = 2.42; csw[7] = 2.42; csw[8] = 2.25; csw[9] = 1.95;
  csw2 = vector(1,9);
  //Interpolation
  if(T >= Tw[0] && T < Tw[9]){
    spline(Tw, csw, 9, ((csw[2] - csw[1])/(Tw[2] - Tw[1])), ((csw[9] - csw[8])/(Tw[9] - Tw[8])), csw2);
    splint(Tw, csw, csw2, 9, T, &csCE);
  }
  else if(T < Tw[1]){
    csCE = csw[1];
  }
  else if(T >= Tw[9]){
    csCE = csw[9];
  }
  Lw = hcl*Getsigv(T, csCE, 1.0)*exp(-exponent/T);
  lossCE = valDens*abund[aC]*xe*Lw;
  //free vectors
  free_vector(powTw,1,9);
  free_vector(Tw,1,9);
  free_vector(csw,1,9);
  free_vector(csw2,1,9);
  return lossCE;
}

double GetLossCH2(double T)
/*Carbon cooling due to collision w H2*/
{
  //Declare Variables
  double hcl, exponent;
  double *Tf, *Lpara, *Lpara2, *Lortho, *Lortho2;
  double Lparafinal, Lorthofinal, lossCH2;
  //Flower Launay 1977 CII -> H2
  hcl = HPLANCK*CSPEED/(157.7E-4);
  exponent = hcl/KB;
  //Temp Table
  Tf = vector(1,17);
  int i;
  for(i=1;i<=17;i++){
    if(i<=10) Tf[i] = 10.0*i;
    else if(i > 10 && i < 17) Tf[i] = Tf[i-1] + 20.0;
    else Tf[i] = 250.0;
  }
  //Para Table (*E24)
  Lpara = vector(1,17);
  Lpara[1] = 8.2E-4; Lpara[2] = 8.9E-2; Lpara[3] = 4.3E-1; Lpara[4] = 9.7E-1; Lpara[5] = 1.6;
  Lpara[6] = 2.2; Lpara[7] = 2.8; Lpara[8] = 3.3; Lpara[9] = 3.9; Lpara[10] = 4.3;
  Lpara[11] = 5.1; Lpara[12] = 5.8; Lpara[13] = 6.4; Lpara[14] = 6.9; Lpara[15] = 7.3;
  Lpara[16] = 7.7; Lpara[17] = 8.2;
  Lpara2 = vector(1,17);
  //Ortho table (*E24)
  Lortho = vector(1,17);
  Lortho[1] = 1.2E-3; Lortho[2] = 1.2E-1; Lortho[3] = 5.7E-1; Lortho[4] = 1.2; Lortho[5] = 2.0;
  Lortho[6] = 2.7; Lortho[7] = 3.4; Lortho[8] = 4.1; Lortho[9] = 4.7; Lortho[10] = 5.2;
  Lortho[11] = 6.2; Lortho[12] = 7.0; Lortho[13] = 7.8; Lortho[14] = 8.4; Lortho[15] = 9.0;
  Lortho[16] = 9.5; Lortho[17] = 1.0E1;
  Lortho2 = vector(1,17);
  //Interpolation
  if(T <= Tf[17]){
    //Ortho
    spline(Tf, Lortho, 17, ((Lortho[2] - Lortho[1])/(Tf[2] - Tf[1])), ((Lortho[17] - Lortho[16])/(Tf[17] - Tf[16])), Lortho2);
    splint(Tf, Lortho, Lortho2, 17, T, &Lorthofinal);
    Lorthofinal *= 1.0E-24;
    //Para
    spline(Tf, Lpara, 17, ((Lpara[2] - Lpara[1])/(Tf[2] - Tf[1])), ((Lpara[17] - Lpara[16])/(Tf[17] - Tf[16])), Lpara2);
    splint(Tf, Lpara, Lpara2, 17, T, &Lparafinal);
    Lparafinal *= 1.0E-24;
  }
  else if(T > Tf[17]){
    Lorthofinal = Lortho[17]*1.0E-24;
    Lparafinal = Lpara[17]*1.0E-24;
  }
  lossCH2 = valDens*abund[aC]*abund[aH]*valxH2*(Lparafinal*0.25 + Lorthofinal*0.75)*exp(-exponent/T);
  //Free vectors
  free_vector(Tf,1,17);
  free_vector(Lpara,1,17);
  free_vector(Lpara2,1,17);
  free_vector(Lortho,1,17);
  free_vector(Lortho2,1,17);
  return lossCH2;
}

double GetLossOxE(double T, double xe, double xO)
/* O -> e*/
{
  //Declare Variables
  double *Tb;
  double hcl21, exp21, *cs21b, *cs21b2, cs21bfinal, Lb21;
  double hcl20, exp20, *cs20b, *cs20b2, cs20bfinal, Lb20;
  double hcl10, exp10, *cs10b, *cs10b2, cs10bfinal, Lb10;
  double lossOxE;
  //Bell 1998 Ox -> e- (Table 2)
  //Table 3 Cooling (Assume Tgas = 100K)
  //Temp Table
  Tb = vector(1,7);
  Tb[1] = 5.0E1; Tb[2] = 1.0E2; Tb[3] = 2.0E2; Tb[4] = 5.0E2; Tb[5] = 1.0E3;
  Tb[6] = 2.0E3; Tb[7] = 3.0E3;
  //Collision Strength 3P2 -> 3P1
  hcl21 = HPLANCK*CSPEED/(63.1E-4);
  exp21 = hcl21/KB;
  cs21b = vector(1,7);
  cs21b[1] = 8.34E-4; cs21b[2] = 1.26E-3; cs21b[3] = 1.79E-3; cs21b[4] = 2.58E-3; cs21b[5] = 3.35E-3;
  cs21b[6] = 4.61E-3; cs21b[7] = 5.92E-3;
  cs21b2 = vector(1,7);
  //Collision Strength 3P2 -> 3P0
  hcl20 = HPLANCK*CSPEED/(44.2E-4);
  exp20= hcl20/KB;
  cs20b = vector(1,7);
  cs20b[1] = 3.23E-4; cs20b[2] = 5.00E-4; cs20b[3] = 7.34E-4; cs20b[4] = 1.12E-3; cs20b[5] = 1.49E-3;
  cs20b[6] = 2.08E-3; cs20b[7] = 2.64E-3;
  cs20b2 = vector(1,7);
  //Collision Strength 3P1 -> 3P0
  hcl10 = HPLANCK*CSPEED/(145.6E-4);
  exp10 = hcl10/KB;
  cs10b = vector(1,7);
  cs10b[1] = 2.74E-7; cs10b[2] = 8.92E-7; cs10b[3] = 2.78E-6; cs10b[4] = 1.05E-5; cs10b[5] = 2.38E-5;
  cs10b[6] = 4.62E-5; cs10b[7] = 6.98E-5;
  cs10b2 = vector(1,7);
  //Interpolation
  if(T < Tb[7] && T >= Tb[0]){
    //cs21
    spline(Tb, cs21b, 7, ((cs21b[2] - cs21b[1])/(Tb[2] - Tb[1])), ((cs21b[7] - cs21b[6])/(Tb[7] - Tb[6])), cs21b2);
    splint(Tb, cs21b, cs21b2, 7, T, &cs21bfinal);
    //cs20
    spline(Tb, cs20b, 7, ((cs20b[2] - cs20b[1])/(Tb[2] - Tb[1])), ((cs20b[7] - cs20b[6])/(Tb[7] - Tb[6])), cs20b2);
    splint(Tb, cs20b, cs20b2, 7, T, &cs20bfinal);
    //cs10
    spline(Tb, cs10b, 7, ((cs10b[2] - cs10b[1])/(Tb[2] - Tb[1])), ((cs10b[7] - cs10b[6])/(Tb[7] - Tb[6])), cs10b2);
    splint(Tb, cs10b, cs10b2, 7, T, &cs10bfinal);
  }
  else if(T >= Tb[7]){
    cs21bfinal = cs21b[7];
    cs20bfinal = cs20b[7];
    cs10bfinal = cs10b[7];
  }
  else if(T < Tb[1]){
    cs21bfinal = cs21b[1];
    cs20bfinal = cs20b[1];
    cs10bfinal = cs10b[1];
  }
  //Cooling Efficiencies in ergs cm3 s-1
  Lb21 = hcl21*Getsigv(T,cs21bfinal,1.0)*exp(-exp21/T);
  Lb20 = hcl20*Getsigv(T,cs20bfinal,1.0)*exp(-exp20/T);
  Lb10 = hcl10*Getsigv(T,cs10bfinal,1.0)*exp(-exp10/T);
  lossOxE = valDens*xe*abund[2]*(1.0-xO)*(Lb21 + Lb20 + Lb10); //ergs s-1
  free_vector(Tb,1,7);
  free_vector(cs21b,1,7);
  free_vector(cs21b2,1,7);
  free_vector(cs20b,1,7);
  free_vector(cs20b2,1,7);
  free_vector(cs10b,1,7);
  free_vector(cs10b2,1,7);
  return lossOxE;
}

double GetLossOxHp(double T, double xO)
/* O -> H+ */
{
  //Declare Variables
  double T2, T3, T4;
  double hcl21, hcl20, hcl10;
  double exp21, exp20, exp10;
  double cs21p, cs20p, cs10p;
  double L21p, L20p, L10p, lossOxHp;
  //Pequignot 1990, 1996 Ox -> H+
  //Eqn 5: cs(T) = a_n*(T/10^n)^b_n
  //Table 3 collision strengths
  T2 = T*1.0E-2;
  T3 = T*1.0E-3;
  T4 = T*1.0E-4;
  //Energies and exponents
  hcl21 = HPLANCK*CSPEED/(63.1E-4);
  hcl20 = HPLANCK*CSPEED/(44.2E-4);
  hcl10 = HPLANCK*CSPEED/(145.6E-4);
  exp21 = hcl21/KB;
  exp20 = hcl20/KB;
  exp10 = hcl10/KB;
  //Collision strength finder
  if(T <= 1.0E2){
    cs21p = 1.40E-3*pow(T2,0.90);
    cs20p = 1.12E-4*pow(T2,1.60);
    cs10p = 3.10E-4*pow(T2,1.06);
  }
  else if(T > 1.0E2 && T <= 1.0E3){
    cs21p = 2.14E-2*pow(T3,1.30);
    cs20p = 3.90E-3*pow(T3,1.40);
    cs10p = 3.10E-4*pow(T3,1.06);
  }
  else if(T > 1.0E3){
    cs21p = 2.78E-1*pow(T4,0.82);
    cs20p = 8.25E-2*pow(T4,0.80);
    cs10p = 2.29E-2*pow(T4,0.69);
  }
  //Cooling Efficiencies ergs cm3 s-1
  L21p = hcl21*Getsigv(T,cs21p,1.0)*exp(-exp21/T);
  L20p = hcl20*Getsigv(T,cs20p,1.0)*exp(-exp20/T);
  L10p = hcl10*Getsigv(T,cs10p,1.0)*exp(-exp10/T);
  lossOxHp = valDens*abund[2]*(1.0-xO)*abund[0]*valxHp*(L21p + L20p + L10p); //ergs s-1
  return lossOxHp;
}

double GetLossOxH(double T, double xO)
/* O -> H */
{
  //Declare Variables
  double T2, xe, xH, lossOxH;
  double GetCollHM(double lam, double gamE, double xe, double gamH, double xH0, double T);
  //OI Fine Structure Line Cooling terms from HM1989 Table 8
  //Loss = nn*abund*x(gamH + gamE) ergs s-1
  //gamH gamE found by GetCollHM(lam,gamE,xe,gamH,x0ion[iH],T) lam in cm
  T2 = T*1.0E-2;
  lossOxH = 0.0;
  xe = 0.0; //O -> e separate function
  xH = GetxH(GetnH2(valxH2),GetnHp(valxHp));
  //OI fine structure
  //lossOxH += valDens*abund[2]*(1.0-xO)*GetCollHM(63.1E-4,0.0,xe,9.2E-11*pow(T2,0.67),xH,T); //2 -> 1
  lossOxH += valDens*abund[2]*(1.0-xO)*GetCollHM(44.2E-4,0.0,xe,4.3E-11*pow(T2,0.80),xH,T); //2 -> 0
  //lossOxH += valDens*abund[2]*(1.0-xO)*GetCollHM(145.6E-4,0.0,xe,1.1E-10*pow(T2,0.44),xH,T); //1 -> 0
  return lossOxH;
}

double GetLossSiH(double T)
/* Si -> H */
{
  //Declare variables
  double hcl, exponent;
  double *Tr, *Lr, *Lr2, Lrfinal, lossSiH;
  //Roueff 1990 SiII -> H
  //Table 2 Temp and L
  hcl = HPLANCK*CSPEED/(34.8E-4);
  exponent = hcl/KB;
  //Temp Table
  Tr = vector(1,12);
  Tr[1] = 1.0E2; Tr[2] = 2.0E2; Tr[3] = 3.0E2; Tr[4] = 4.0E2; Tr[5] = 5.0E2;
  Tr[6] = 6.0E2; Tr[7] = 7.0E2; Tr[8] = 8.0E2; Tr[9] = 9.0E2; Tr[10] = 1.0E3;
  Tr[11] = 1.5E3; Tr[12] = 2.0E3;
  //L Table (*E24)
  Lr = vector(1,12);
  Lr[1] = 9.0E-1; Lr[2] = 7.9; Lr[3] = 17.3; Lr[4] = 26.4; Lr[5] = 34.6;
  Lr[6] = 41.9; Lr[7] = 48.6; Lr[8] = 54.6; Lr[9] = 60.0; Lr[10] = 65.0;
  Lr[11] = 86.0; Lr[12] = 102.0;
  Lr2 = vector(1,12);
  //Interpolation
  if(T >= Tr[1] && T < Tr[12]){
    spline(Tr, Lr, 12, ((Lr[2] - Lr[1])/(Tr[2] - Tr[1])), ((Lr[12] - Lr[11])/(Tr[12] - Tr[11])), Lr2);
    splint(Tr, Lr, Lr2, 12, T, &Lrfinal);
  }
  else if(T < Tr[1]) Lrfinal = Lr[1];
  else if(T >= Tr[12]) Lrfinal = Lr[12];
  lossSiH = (1.0E-24)*valDens*abund[4]*abund[0]*GetxH(GetnH2(valxH2),GetnHp(valxHp))*Lrfinal*exp(-exponent/T);
  free_vector(Tr,1,12);
  free_vector(Lr,1,12);
  free_vector(Lr2,1,12);
  return lossSiH;
}

double GetLossSiE(double T, double xe)
/* Si -> e */
{
  //Declare Variables
  double hcl, exponent;
  double *powTd, *Td, *csd, *csd2, csdfinal, Ld, lossSiE;
  //Dufton 1991 Si -> e-
  //Table 1 P 1/2 -> P 3/2
  hcl = HPLANCK*CSPEED/(34.8E-4);
  exponent = hcl/KB;
  //Temp Table
  powTd = vector(1,6);
  powTd[1] = 3.6; powTd[2] = 3.8; powTd[3] = 4.0; powTd[4] = 4.2; powTd[5] = 4.4;
  powTd[6] = 4.6;
  Td = vector(1,6);
  int i;
  for(i = 1; i <= 6; i++){
    Td[i] = pow(10.0,powTd[i]);
  }
  //Collision Strength
  csd = vector(1,6);
  csd[1] = 5.58; csd[2] = 5.61; csd[3] = 5.70; csd[4] = 5.79; csd[5] = 5.75;
  csd[6] = 5.47;
  csd2 = vector(1,6);
  //Interpolation
  if(T >= Td[1] && T < Td[6]){
    spline(Td, csd, 6, ((csd[2] - csd[1])/(Td[2] - Td[1])), ((csd[6] - csd[5])/(Td[6] - Td[5])), csd2);
    splint(Td, csd, csd2, 6, T, &csdfinal);
  }
  else if(T < Td[1]) csdfinal = csd[1];
  else if(T >= Td[6]) csdfinal = csd[6];
  Ld = hcl*Getsigv(T,csdfinal,1.0)*exp(-exponent/T);
  lossSiE = valDens*abund[4]*xe*Ld; //ergs s-1
  free_vector(powTd,1,6);
  free_vector(Td,1,6);
  free_vector(csd,1,6);
  free_vector(csd2,1,6);
  return lossSiE;
}

double GetLossHE(double T, double xe)
/* H -> e */
{
  //Declare Variables
  double *Ts, *Ls, *Ls2, Lsfinal, lossHE;
  //Spitzer 1978 (Taken from DM1972)
  // electron impact HI
  //Temp Table
  Ts = vector(1,16);
  Ts[1] = 3.0E3; Ts[2] = 4.0E3; Ts[3] = 5.0E3; Ts[4] = 5.5E3; Ts[5] = 6.0E3;
  Ts[6] = 6.5E3; Ts[7] = 7.0E3; Ts[8] = 7.5E3; Ts[9] = 8.0E3; Ts[10] = 8.5E3;
  Ts[11] = 9.0E3; Ts[12] = 9.5E3; Ts[13] = 1.0E4; Ts[14] = 1.05E4; Ts[15] = 1.1E4;
  Ts[16] = 1.15E4;
  //Cooling Efficiency (*E24)
  Ls = vector(1,16);
  Ls[1] = 5.3E-12; Ls[2] = 1.0E-7; Ls[3] = 3.7E-5; Ls[4] = 3.2E-4; Ls[5] = 1.9E-3;
  Ls[6] = 8.8E-3; Ls[7] = 3.2E-2; Ls[8] = 1.0E-1; Ls[9] = 2.7E-1; Ls[10] - 6.5E-1;
  Ls[11] = 1.4E0; Ls[12] = 2.9E0; Ls[13] = 5.4E0; Ls[14] = 9.5E0; Ls[15] = 1.59E1;
  Ls[16] = 2.56E1;
  Ls2 = vector(1,16);
  //Interpolation
  if(T >= Ts[1] && T < Ts[16]){
    spline(Ts, Ls, 16, ((Ls[2] - Ls[1])/(Ts[2] - Ts[1])), ((Ls[16] - Ls[15])/(Ts[16] - Ts[15])), Ls2);
    splint(Ts, Ls, Ls2, 16, T, &Lsfinal);
  }
  else if(T < Ts[1]) Lsfinal = Ls[1];
  else if(T >= Ts[16]) Lsfinal = Ls[16];
  lossHE = 1.0E-24*valDens*xe*abund[0]*GetxH(GetnH2(valxH2),GetnHp(valxHp))*Lsfinal; //ergs s-1
  free_vector(Ts,1,16);
  free_vector(Ls,1,16);
  free_vector(Ls2,1,16);
  return lossHE;
}

double GetCollHM(double lam, double gamE, double xe, double gamH, double xH0, double T){
  //Declare Variables
  double hcl, exponent, CollHM;
  //Calculates collisional rates given by HM 1989 cm3 s-1
  hcl = HPLANCK*CSPEED/lam;
  exponent = -hcl/KB;
  CollHM = hcl*(xH0*gamH + xe*gamE)*exp(exponent/T);
  return CollHM;
}

double GetLossRec(double T, double xe)
/* Radiative recombination of e w small grains and PAHs Wolfire(2003)*/
{
  double psi, beta, lossRec;
  psi = (CHI*sqrt(T))/(valDens*xe);
  beta = 0.74/pow(T,0.068);
  lossRec = (4.65E-30)*valDens*xe*pow(T,0.94)*pow(psi,beta);
  return lossRec;
}

/*=============================================================================
 * END GETLOSS FUNCTION HELPER FUNCTIONS
 *---------------------------------------------------------------------------*/

static double GetGain(double T, double xe)
/*Net heating terms in ergs s-1*/
{
  double gainPE, gainCR, gainH2, gaintot;
  double GetGainPE(double T, double xe);
  double GetGainCR();
  double GetGainH2grain(double T);
  gainPE = GetGainPE(T,xe); //Photoelectric effect
  gainCR = GetGainCR(); //Cosmic ray heating
  gainH2 = GetGainH2grain(T); //H2 formation on dust grains
  gaintot = gainPE + gainCR + gainH2; 
  return gaintot;
}

/*=============================================================================
 * GETGAIN FUNCTION HELPER FUNCTIONS
 * GetGainPE()        - Heating due to photoelectric effect
 * GetGainCR()        - Heating due to cosmic rays
 * GetGainH2grain()   - Heating due to H2 grain formation
 *----------------------------------------------------------------------------*/

double GetGainPE(double T, double xe)
/*Photoelectric effect Wolfire 2003*/
{
  double T4, psi, denom1, denom2, num2, eps, gainPE;
  T4 = 1.0E-4*T;
  psi = (CHI*sqrt(T))/(xe*valDens);
  denom1 = 1.0+(4.0E-3)*pow(psi,0.73);
  denom2 = 1.0+(2.0E-4)*psi;
  num2 = (3.7E-2)*pow(T4,0.7);
  eps = ((4.9E-2)/denom1) + (num2/denom2);
  gainPE = 1.3E-24*eps*CHI;
  return gainPE;
}

double GetGainCR()
/*Cosmic Ray Heating: Goldsmith and Langer 1978*/
{
  double gainCR;
  gainCR = 3.2E-11*ZETA;
  return gainCR;
}

double GetGainH2grain(double T)
{
  double RH2, xH, logTemp, ncrH, ncrH2, ncr, gainH2;
  RH2 = k1(T,GRAIN_TEMP);
  xH = GetxH(GetnH2(valxH2),GetnHp(valxHp));
  logTemp = log10(1.0E-4*T);
  ncrH = pow(10.0,3.0-0.416*logTemp-0.327*logTemp*logTemp);
  ncrH2 = pow(10.0,4.845-1.3*logTemp+1.62*logTemp*logTemp);
  ncr = (ncrH*ncrH2)/(valxH2*ncrH + xH *ncrH2);
  gainH2 = ((7.2E-12)*RH2)/(valDens+ncr);
  return gainH2;
}

/*=============================================================================
 * END GETGAIN FUNCTION HELPER FUNCTIONS
 *---------------------------------------------------------------------------*/

static void spline(double x[], double y[], int n, double yp1, double ypn, double y2[])
/*Given arrays x[1..n] and y[1..n] containing a tabulated function, i.e, y1 = f(xi), with
  x1 < x2 < ..<xN, and given values yp1 and ypn for the first derivative of the interpolating 
  function at points 1 and n, respectively, this routine returns an array y2[1..n] that contains
  the second derivatives of the interpolating function at the tabulated points xi. If yp1 and/or
  ypn are equal to 1*10^30 or larger, the routine is signaled to set the corresponding boundary
  condition for a natural spline, with zero second derivative on that boundary
*/
{
  int i,k;
  double p,qn,sig,un,*u;
  u = vector(1,n-1);
  if(yp1 > 0.99E30) //The lower boundary condition is set either to be "natural" or else to have a specified first derivative
    y2[1] = u[1] = 0.0;
  else{
    y2[1] = -0.5;
    u[1] = (3.0/(x[2] - x[1]))*((y[2] - y[1])/(x[2] - x[1]) - yp1);
  }
  for(i=2;i<=n-1;i++){
    /*This is the decomposition loop of the tridiagonal algorithm.
      y2 and u are used for temporary storage of the decomposed factors
    */
    sig = (x[i] - x[i-1])/(x[i+1] - x[i-1]);
    p = sig*y2[i-1]+2.0;
    y2[i] = (sig-1.0)/p;
    u[i] = (y[i+1] - y[i])/(x[i+1] - x[i]) - (y[i] - y[i-1])/(x[i] - x[i-1]);
    u[i] = (6.0*u[i]/(x[i+1] - x[i-1])-sig*u[i-1])/p;
  }
  if(ypn > 0.99E30) //Upper BC set to "natural"
    qn = un = 0.0;
  else{
    qn = 0.5;
    un = (3.0/(x[n] - x[n-1]))*(ypn - (y[n] - y[n-1])/(x[n]- x[n-1]));
  }
  y2[n] = (un-qn*u[n-1])/(qn*y2[n-1]+1.0);
  for(k=n-1;k>=1;k--) //This is the bksub loop of tridiagonal algorithm
    y2[k] = y2[k]*y2[k+1]+u[k];
  free_vector(u,1,n-1);
}

static void splint(double xa[], double ya[], double y2a[], int n, double x, double *y)
/*Given the arrays xa[1..n] and ya[1..n], which tabulate a function (with the xai's in order),
  and given the array y2a[1..n], which is the ouput from spline above, and given a value of 
  x, this routine returns a cubic-spline interpolated value y
*/
{
  int klo,khi,k;
  double h,b,a;
  klo = 1;
  /*We will find the right place in the table by means of 
    bisection. This is optimal if sequential call to this
    routine are at random values of x. If sequential calls
    are in order, and closely spaced, one would do better
    to store previous values of klo and khi and test if
    they remain appropriate on the next call
  */
  khi = n;
  while (khi - klo > 1){
    k = (khi+klo) >> 1;
    if(xa[k] > x) khi = k;
    else klo = k;
  } //klo and khi bracket the input value of x
  h = xa[khi] - xa[klo];
  if(h == 0.0) nrerror("Bad xa input to routine splint"); // the xa's must be distinct
  a = (xa[khi] - x)/h;
  b = (x - xa[klo])/h; //Cubic spline polynomial is now evaluated
  *y = a*ya[klo] + b*ya[khi] + ((a*a*a-a)*y2a[klo] + (b*b*b-b)*y2a[khi])*(h*h)/6.0;
}

static double Getsigv(double Temp, double cs, double degen)
{
  double T4, sigv;
  T4 = Temp*1.0E-4;
  sigv = ((8.629E-8)*cs)/(sqrt(T4)*degen);
  return sigv;
}

//INTEGRATION FUNCTIONS

//ODE INTEGRATOR

// User storage for intermediate results. Preset kmax and dxsav in
// the calling program. If kmax != 0 results are stored at approximate
// intervals dxsav in the arrays xp[1..kount], yp[1..kount][1..nvar],
// where kount is output by odeint. Defining declarations for these
// variables, with memory allocations xp[1..kmax] and
// yp[1..kmax][1..nvar] for the arrays, should be in the calling program
static void odeint(double ystart[], int nvar, double x1, double x2, double eps,
		   double h1, double hmin, int *nok, int *nbad,int boolModel, 
		   void (*derivs)(double, int, double[], double[]),
		   void (*rkqs)(double[], double[], int, double*, double, double, double[],
				double*, double*, void (*)(double, int, double[], double[])))  
{
  /* Driver with adaptive stepsize control. Integrate starting values ystart[1..nvar] from x1 to x2 
  with accuracy eps, storing intermediate results in global variables. h1 should be set as a guessed 
  first stepsize, hmin as the minimum allowed stepsize (can be zero). On output nok and nbad are the 
  number of good and bad (but retried and fixed) steps taken, and ystart is replaced by values at 
  the end of the integration interval. derivs is the user-supplied routine for calculating the right-hand 
  side derivative,*/

  int nstp,i;
  double xsav,x,hnext,hdid,h;
  double *yscal,*y,*dydx;

  yscal=vector(1,nvar);
  y=vector(1,nvar);
  dydx=vector(1,nvar);
  x=x1;
  h=SGN(h1,x2-x1);
  *nok = (*nbad) = kount = 0;

  for (i=1;i<=nvar;i++) y[i]=ystart[i];
  
  if(kmax > 0) xsav=x-dxsav*2.0; //Assures storage of first step

  for(nstp=1;nstp<=MAXSTPS;nstp++){ //Take at most MAXSTP steps
    (*derivs)(x,nvar,y,dydx);

    for(i=1;i<=nvar;i++) //Scaling used to monitor accuracy. This general-purpose choice
	                       //can be modified if need be
      yscal[i]=fabs(y[i])+fabs(dydx[i]*h)+TINY;
	 
    if(kmax > 0 && kount < kmax-1 && fabs(x-xsav) > fabs(dxsav)) {
      xp[++kount]=x;  //Stores intermediate results.

      for(i=1;i<=nvar;i++) yp[i][kount]=y[i];
      xsav=x;
    }
	  
    if((x+h-x2)*(x+h-x1) > 0.0) h=x2-x; //If stepsize can overshoot, decrease
    (*rkqs)(y,dydx,nvar,&x,h,eps,yscal,&hdid,&hnext,derivs);

    if(hdid==h) ++(*nok);
    else ++(*nbad);

    if((x-x2)*(x2-x1) >= 0.0){  //Are we done?
      for(i=1;i<=nvar;i++) ystart[i]=y[i];
      if(kmax){
	xp[++kount]=x;
	for(i=1;i<=nvar;i++) yp[i][kount]=y[i];
      }
      free_vector(dydx,1,nvar);
      free_vector(y,1,nvar);
      free_vector(yscal,1,nvar);
      return;                     //Normal Exit!
    }
	  
    if(fabs(hnext) <= hmin) nrerror("Step size too small in odeint");
    h=hnext;
    if(boolModel == 1){
      hvalueCN = h;
    }
    else{
      hvalueT = h;
    }
 
  }
  nrerror("Too many steps in routine odeint");
}

//STIFF INTEGRATOR
#define MAXTRY 80 // default 40
#define SAFETY 0.9
#define GROW 1.5
#define PGROW -0.25
#define SHRNK 0.5
#define PSHRNK (-1.0/3.0)
#define ERRCON 0.1296
// Here NMAX is the maximum value of n; GROW and SHRNK are the largest and smallest factors
// by which stepsize can change in one step; ERRCON equals (GROW/SAFETY) raised to the power
// (1/PGROW) and handles the case when errmax ~= 0.
#define GAM 0.231
#define A21 2.0
#define A31 4.52470820736
#define A32 4.16352878860
#define C21 -5.07167533877
#define C31 6.02015272865
#define C32 0.159750684673
#define C41 -1.856343618677
#define C42 -8.50538085819
#define C43 -2.08407513602
#define B1 3.95750374663
#define B2 4.62489238836
#define B3 0.617477263873
#define B4 1.282612945268
#define E1 -2.30215540292
#define E2 -3.07363448539
#define E3 0.873280801802
#define E4 1.282612945268

#define C1X GAM
#define C2X -0.396296677520e-01
#define C3X 0.550778939579
#define C4X -0.553509845700e-01
#define A2X 0.462
#define A3X 0.880208333333
static void stiff(double y[], double dydx[], int n, double *x, double htry, double eps,
		  double yscal[], double *hdid, double *hnext, void (*derivs)(double, int, double[], double[]))
// Fourth-order Rosenbrock step for integrating stiffo.d.e.s, with monitoring of local truncation
// error to adjust stepsize. Input are the dependent variable vector y[1..n] and its derivative
// dydx[1..n] at the starting value of the independent variable x. Also input are the stepsize to
// be attempted htry, the required accuracy eps, and the vector yscal[1..n] against which
// the error is scaled. On output, y and x are replaced by their new values, hdid is the stepsize
// that was actually accomplished, and hnext is the estimated next stepsize. derivs is a usersupplied
// routine that computes the derivatives of the right-hand side with respect to x, while
// jacobn (a fixed name) is a user-supplied routine that computes the Jacobi matrix of derivatives
// of the right-hand side with respect to the components of y.
{
  void jacobn(double x, double y[], double dfdx[], double **dfdy, int n);
  void lubksb(double **a, int n, int *indx, double b[]);
  void ludcmp(double **a, int n, int *indx, double *d);
  int i,j,jtry,*indx;
  double d,errmax,h,xsav,**a,*dfdx,**dfdy,*dysav,*err;
  double *g1,*g2,*g3,*g4,*ysav;
  int boolStiff = 1;

  indx=ivector(1,n);
  a=matrix(1,n,1,n);
  dfdx=vector(1,n);
  dfdy=matrix(1,n,1,n);
  dysav=vector(1,n);
  err=vector(1,n);
  g1=vector(1,n);
  g2=vector(1,n);
  g3=vector(1,n);
  g4=vector(1,n);
  ysav=vector(1,n);
  xsav=(*x); // Save initial values.
  for (i=1;i<=n;i++) {
    ysav[i]=y[i];
    dysav[i]=dydx[i];
  }
  jacobn(xsav,ysav,dfdx,dfdy,n);
  // The user must supply this routine to return the n-by-n matrix dfdy and the vector dfdx.
  h=htry; // Set stepsize to the initial trial value.
  for (jtry=1;jtry<=MAXTRY;jtry++) {
    for (i=1;i<=n;i++) { // Set up the matrix 1 - gamma*h*f'
      for (j=1;j<=n;j++) a[i][j] = -dfdy[i][j];
      a[i][i] += 1.0/(GAM*h);
    }
    ludcmp(a,n,indx,&d); // LU decomposition of the matrix.
    for (i=1;i<=n;i++) // Set up right-hand side for g1.
      g1[i]=dysav[i]+h*C1X*dfdx[i];
		
    lubksb(a,n,indx,g1); // Solve for g1.
    for (i=1;i<=n;i++) // Compute intermediate values of y and x.
      y[i]=ysav[i]+A21*g1[i];

    //Check if Abundance < 0
    for(i=1;i<=n;i++){
      if(y[i] < 0.0)
	boolStiff = 0;
    }

    if(boolStiff == 1){
      *x=xsav+A2X*h;
      (*derivs)(*x,n,y,dydx); // Compute dydx at the intermediate values.
      for (i=1;i<=n;i++) // Set up right-hand side for g2.
	g2[i]=dydx[i]+h*C2X*dfdx[i]+C21*g1[i]/h;
      lubksb(a,n,indx,g2); // Solve for g2.
      for (i=1;i<=n;i++) // Compute intermediate values of y and x.
	y[i]=ysav[i]+A31*g1[i]+A32*g2[i];

      //Check if Abundance < 0.0
      for(i=1;i<=n;i++){
	if(y[i] < 0.0)
	  boolStiff = 0;
      }
    }
    
    if(boolStiff == 1){
      *x=xsav+A3X*h;
      (*derivs)(*x,n,y,dydx); // Compute dydx at the intermediate values.
      for (i=1;i<=n;i++) // Set up right-hand side for g3.
	g3[i]=dydx[i]+h*C3X*dfdx[i]+(C31*g1[i]+C32*g2[i])/h;
      lubksb(a,n,indx,g3); // Solve for g3.
      
      for (i=1;i<=n;i++) // Set up right-hand side for g4.
	g4[i]=dydx[i]+h*C4X*dfdx[i]+(C41*g1[i]+C42*g2[i]+C43*g3[i])/h;
      lubksb(a,n,indx,g4); // Solve for g4.
      
      for (i=1;i<=n;i++) { // Get fourth-order estimate of y and error estimate.
	y[i]=ysav[i]+B1*g1[i]+B2*g2[i]+B3*g3[i]+B4*g4[i];
	err[i]=E1*g1[i]+E2*g2[i]+E3*g3[i]+E4*g4[i];
      }
      
      //Check if Abundance < 0.0
      for(i=1;i<=n;i++){
	if(y[i] < 0.0)
	  boolStiff = 0;
      }
    }

    if(boolStiff == 1){
      *x=xsav+h;
      if (*x == xsav) nrerror("stepsize not significant in stiff");
      errmax=0.0; // Evaluate accuracy.
      for (i=1;i<=n;i++) errmax=FMAX(errmax,fabs(err[i]/yscal[i]));
      
      errmax /= eps; // Scale relative to required tolerance.

      //END IN CASE
      if (errmax <= 1.0) { // Step succeeded. Compute size of next step and return
	*hdid=h;
	*hnext=(errmax > ERRCON ? SAFETY*h*pow(errmax,PGROW) : GROW*h);
	free_vector(ysav,1,n);
	free_vector(g4,1,n);
	free_vector(g3,1,n);
	free_vector(g2,1,n);
	free_vector(g1,1,n);
	free_vector(err,1,n);
	free_vector(dysav,1,n);
	free_matrix(dfdy,1,n,1,n);
	free_vector(dfdx,1,n);
	free_matrix(a,1,n,1,n);
	free_ivector(indx,1,n);
	return;
      } else { // Truncation error too large, reduce step size
	*hnext=SAFETY*h*pow(errmax,PSHRNK);
	h=(h >= 0.0 ? FMAX(*hnext,SHRNK*h) : FMIN(*hnext,SHRNK*h));
      }
    }
    
    if(boolStiff == 0){
      printf("Abundance Value < 0.0! xH2 = %.3e\txHp = %.3e\thnext = %.3e\tReduce Stepsize!\n",GetxH2(y[1]),GetxHp(y[2]),h);
      printf("xpos = %.5e\typos = %.5e\tzpos = %.5e\n",xpos,ypos,zpos);
      for(i=1;i<=n;i++)
	y[i] = ysav[i];
      *x = xsav;
      h *= 0.5;
      *hnext = h;
      errmax = 100.0;
      boolStiff = 1;
      if(*hnext < 1.0E-1) nrerror("Stiff: Stepsize too small!\n");
    }

  } // Go back and re-try step
  nrerror("exceeded MAXTRY in stiff");
}

/*=============================================================================
 * STIFF FUNCTION HELPER FUNCTIONS
 * jacobn()           - Calculates jacobian of dnHxdt
 * ludcmp()           - LU Decomposition
 * lubksb()           - LU Back Substitution
 *----------------------------------------------------------------------------*/
void jacobn(double x, double y[], double dfdx[], double **dfdy, int n)
{
  int i,j;
  double h,h2,original;
  double *forward, *backward;
  forward = vector(1,n);
  backward = vector(1,n);
  
  for(i=1;i<=n;i++){
    dfdx[i] = 0.0;
    for(j=1;j<=n;j++){
      h = 1.0E-4*y[j];
      h2 = 2.0*h;
      original = y[j];
      y[j] = original + h;
      dnHxdt(x,n,y,forward);
      y[j] = original - h;
      dnHxdt(x,n,y,backward);
      y[j] = original;
      dfdy[i][j] = (forward[i] - backward[i])/h2;
    }
  }
  free_vector(forward,1,n);
  free_vector(backward,1,n);

}

void lubksb(double **a, int n, int *indx, double b[])
{
  int i,ii=0,ip,j;
  double sum;

  for (i=1;i<=n;i++) {
    ip=indx[i];
    sum=b[ip];
    b[ip]=b[i];
    if (ii)
      for (j=ii;j<=i-1;j++) sum -= a[i][j]*b[j];
    else if (sum) ii=i;
    b[i]=sum;
  }
  for (i=n;i>=1;i--) {
    sum=b[i];
    for (j=i+1;j<=n;j++) sum -= a[i][j]*b[j];
    b[i]=sum/a[i][i];
  }
}

void ludcmp(double **a, int n, int *indx, double *d)
{
  int i,imax,j,k;
  double big,dum,sum,temp;
  double *vv;

  vv=vector(1,n);
  *d=1.0;    //No row interchanges yet
  for (i=1;i<=n;i++) {   //Loop over rows to get implicit scaling info
    big=0.0;
    for (j=1;j<=n;j++)
      if ((temp=fabs(a[i][j])) > big) big=temp;
    if (big == 0.0){
      for(i=1;i<=n;i++){
	for(j=1;j<=n;j++){
	  printf("a[%i][%i] = %.3e\n",i,j,a[i][j]);
	}
      }
      nrerror("Singular matrix in routine ludcmp");
    }
    //Nonzero largest element
    vv[i]=1.0/big;  //Save the scaling
  }
  for (j=1;j<=n;j++) {  //Loop over columns of Crout's method: Eqn(2.3.12) (i=j)
    for (i=1;i<j;i++) {  
      sum=a[i][j];
      for (k=1;k<i;k++) sum -= a[i][k]*a[k][j];
      a[i][j]=sum;
    }
    big=0.0; //Initialize for te search for largest pivot element
    for (i=j;i<=n;i++) { // i=j of equation (2.3.12) i=j+1...N
      sum=a[i][j];
      for (k=1;k<j;k++)
	sum -= a[i][k]*a[k][j];
      a[i][j]=sum;
      if ( (dum=vv[i]*fabs(sum)) >= big) {
	//Is the figure of merit for the pivot better than the best so far?
	big=dum;
	imax=i;
      }
    }
    if (j != imax) {  //Do we need to interchange rows
      for (k=1;k<=n;k++) { //Yes do so
	dum=a[imax][k];
	a[imax][k]=a[j][k];
	a[j][k]=dum;
      }
      *d = -(*d);  //And change parity of d
      vv[imax]=vv[j]; //Also interchange the scale factor
    }
    indx[j]=imax;
    if (a[j][j] == 0.0) a[j][j]=TINY;
    if (j != n) {
      dum=1.0/(a[j][j]);
      for (i=j+1;i<=n;i++) a[i][j] *= dum;
    }
  }
  free_vector(vv,1,n);
}

/*=============================================================================
 * END STIFF FUNCTION HELPER FUNCTIONS
 *---------------------------------------------------------------------------*/

//TM INTEGRATORS

#define SAFETYT 0.9
#define PGROWT -0.2
#define PSHRNKT -0.25
#define ERRCONT 1.89E-4
//The value ERRCON = (5/SAFETYT)**(1/PGROWT)
static void RKQS(double y[], double dydx[], int n, double *x, double htry, double eps, 
		 double yscal[], double *hdid, double *hnext, void (*derivs)(double, int, double [], double []))
/*Fifth-order Runge-Kutta step with monitoring of local truncation error to ensure accuracy and 
  adjust stepsize. Input are the dependent variable vector y[1..n] and its derivative dydx[1..n]
  at the starting value of the independent variable x. Also input are the stepsize to be attempted
  htry, the required accuracy eps, and the vector yscal[1..n] against which the error is 
  scaled. On output, y and x are replaced by their new values. hdid is the stepsize that was 
  actually accomplished, and hnext is the estimated next stepsize. derivs is the user supplied 
  routine that computes the right hand side derivatives
*/
{
  void RKCK(double y[], double dydx[], int n, double x, double h,
	    double yout[], double yerr[], void (*derivs)(double, int, double [], double []));
  int i;
  double errmax, h, htemp, xnew, *yerr, *ytemp;

  yerr = vector(1,n);
  ytemp = vector(1,n);
  h = htry; //Set stepsize to the initial trial value
  for(;;){
    RKCK(y,dydx,n,*x,h,ytemp,yerr,derivs); //Take a step
    //Error Condition
    if(ytemp[1] == 0.0){
      //printf("RKQS: Negative Temperature. Reducing stepsize by half!\th = %.3e\n",h);
      *hdid = h;
      *hnext = 0.5*h;
      if(*hnext < 1.0E-1) nrerror("RKQS Stepsize too small!");
      free_vector(ytemp,1,n);
      free_vector(yerr,1,n);
      return;
    }
    errmax = 0.0; //Evaluate accuracy
    for(i=1;i<=n;i++) errmax = FMAX(errmax,fabs(yerr[i]/yscal[i]));
    errmax /= eps; //Scale relative to required tolerance
    if(errmax <= 1.0) break; //Step succeeded! Compute size of next step
    htemp = SAFETYT*h*pow(errmax,PSHRNKT);
    //Truncation error too large, reduce stepsize
    h = (h >= 0.0 ? FMAX(htemp,0.1*h) : FMIN(htemp,0.1*h));
    //No more than a factor of 10
    xnew = (*x)+h;
    if(xnew == *x){
      printf("RKQS: h = %.3e \t ytemp = %.3e \n",h,ytemp[1]);
      nrerror("Stepsize underflow in rkqs");
    }
  }
  if(errmax > ERRCONT) *hnext = SAFETYT*h*pow(errmax,PGROWT);
  else *hnext = 5.0*h; //No more than a factor of 5 increase
  *x += (*hdid=h);
  for(i=1;i<=n;i++) y[i] = ytemp[i];
  free_vector(ytemp,1,n);
  free_vector(yerr,1,n);
}

void RKCK(double y[], double dydx[], int n, double x, double h, double yout[],
	  double yerr[], void (*derivs)(double, int, double[], double[]))
/*Given the values for n variables y[1..n] and their derivatives dydx[1..n] known at x, use 
  the fifth-order Cash-Karp Runge-Kutta method to advance the solution over an interval h
  and return the incremented variables as yout[1..n]. Also return an estimate of the local
  truncation error in yout using the embedded fourth order method. The user supplies the routine
  derivs(x,y,dydx), which returns derivatives dydx at x
*/
{
  int i;
  static double a2 = 0.2, a3 = 0.3, a4 = 0.6, a5 = 1.0, a6 = 0.875, b21 = 0.2,
    b31 = 3.0/40.0, b32 = 9.0/40.0, b41 = 0.3, b42 = -0.9, b43 = 1.2,
    b51 = -11.0/54.0, b52 = 2.5, b53 = -70.0/27.0, b54 = 35.0/27.0,
    b61 = 1631.0/55296.0, b62 = 175.0/512.0, b63 = 575.0/13824.0,
    b64 = 44275.0/110592.0, b65 = 253.0/4096.0, c1 = 37.0/378.0,
    c3 = 250.0/621.0, c4 = 125.0/594.0, c6 = 512.0/1771.0,
    dc5 = -277.00/14336.0;
  double dc1 = c1 - 2825.0/27648.0, dc3 = c3 - 18575.0/48384.0,
    dc4 = c4 - 13525.0/55296.0, dc6 = c6 - 0.25;
  double *ak2, *ak3, *ak4, *ak5, *ak6, *ytemp;

  ak2 = vector(1,n);
  ak3 = vector(1,n);
  ak4 = vector(1,n);
  ak5 = vector(1,n);
  ak6 = vector(1,n);
  ytemp = vector(1,n);

  for(i=1;i<=n;i++) //First step
    ytemp[i] = y[i] + b21*h*dydx[i];
  if(ytemp[1] <= 0.0){
    //printf("RKCK: Step 1: Negative Temperature. Reduce Stepsize! temp = %.3e\n",y[1]);
    yout[1] = 0.0;
    free_vector(ytemp,1,n);
    free_vector(ak6,1,n);
    free_vector(ak5,1,n);
    free_vector(ak4,1,n);
    free_vector(ak3,1,n);
    free_vector(ak2,1,n);
    return;
  }
  (*derivs)(x+a2*h,n,ytemp,ak2); //Second step
  for(i=1;i<=n;i++)
    ytemp[i] = y[i] + h*(b31*dydx[i] + b32*ak2[i]);
  if(ytemp[1] <= 0.0){
    //printf("RKCK: Step 2: Negative Temperature. Reduce Stepsize! temp = %.3e\n",y[1]);
    yout[1] = 0.0;
    free_vector(ytemp,1,n);
    free_vector(ak6,1,n);
    free_vector(ak5,1,n);
    free_vector(ak4,1,n);
    free_vector(ak3,1,n);
    free_vector(ak2,1,n);
    return;
  }
  (*derivs)(x+a3*h,n,ytemp,ak3); //Third step
  for(i=1;i<=n;i++)
    ytemp[i] = y[i] + h*(b41*dydx[i] + b42*ak2[i] + b43*ak3[i]);
  if(ytemp[1] <= 0.0){
    //printf("RKCK: Step 3: Negative Temperature. Reduce Stepsize! temp = %.3e\n",y[1]);
    yout[1] = 0.0;
    free_vector(ytemp,1,n);
    free_vector(ak6,1,n);
    free_vector(ak5,1,n);
    free_vector(ak4,1,n);
    free_vector(ak3,1,n);
    free_vector(ak2,1,n);
    return;
  }
  (*derivs)(x+a4*h,n,ytemp,ak4); //Fourth step
  for(i=1;i<=n;i++)
    ytemp[i] = y[i] + h*(b51*dydx[i] + b52*ak2[i] + b53*ak3[i] + b54*ak4[i]);
  if(ytemp[1] <= 0.0){
    //printf("RKCK: Step 4: Negative Temperature. Reduce Stepsize! temp = %.3e\n",y[1]);
    yout[1] = 0.0;
    free_vector(ytemp,1,n);
    free_vector(ak6,1,n);
    free_vector(ak5,1,n);
    free_vector(ak4,1,n);
    free_vector(ak3,1,n);
    free_vector(ak2,1,n);
    return;
  }
  (*derivs)(x+a5*h,n,ytemp,ak5); //Fifth step
  for(i=1;i<=n;i++)
    ytemp[i] = y[i] + h*(b61*dydx[i] + b62*ak2[i] + b63*ak3[i] + b64*ak4[i] + b65*ak5[i]);
  if(ytemp[1] <= 0.0){
    //printf("RKCK: Step 5: Negative Temperature. Reduce Stepsize! temp = %.3e\n",y[1]);
    yout[1] = 0.0;
    free_vector(ytemp,1,n);
    free_vector(ak6,1,n);
    free_vector(ak5,1,n);
    free_vector(ak4,1,n);
    free_vector(ak3,1,n);
    free_vector(ak2,1,n);
    return;
  }
  (*derivs)(x+a6*h,n,ytemp,ak6); //Sixth step
  for(i=1;i<=n;i++) //Accumulate increments with proper weights
    yout[i] = y[i] + h*(c1*dydx[1] + c3*ak3[i] + c4*ak4[i] + c6*ak6[i]);
  if(yout[1] <= 0.0){
    //printf("RKCK: Step 6: Negative Temperature. Reduce Stepsize! temp = %.3e\n",ytemp[1]);
    yout[1] = 0.0;
    free_vector(ytemp,1,n);
    free_vector(ak6,1,n);
    free_vector(ak5,1,n);
    free_vector(ak4,1,n);
    free_vector(ak3,1,n);
    free_vector(ak2,1,n);
    return;
  }
  for(i=1;i<=n;i++)
    yerr[i] = h*(dc1*dydx[i] + dc3*ak3[i] + dc4*ak4[i] + dc5*ak5[i] + dc6*ak6[i]);
  //Estimate error as difference between fourth and fifth order methods
  free_vector(ytemp,1,n);
  free_vector(ak6,1,n);
  free_vector(ak5,1,n);
  free_vector(ak4,1,n);
  free_vector(ak3,1,n);
  free_vector(ak2,1,n);
}

/*==============================================================================
 * SELF GRAVITY FUNCTIONS
 * GetMeanRho()       - Calculates the global variable grav_mean_rho for MPI runs
 *-----------------------------------------------------------------------------*/

#ifdef SELF_GRAVITY
static void GetMeanRho(DomainS *pDomain)
{
  GridS *pG = (pDomain -> Grid);
  int i, is = pG -> is, ie = pG -> ie;
  int j, js = pG -> js, je = pG -> je;
  int k, ks = pG -> ks, ke = pG -> ke;
  Real Nx1 = pDomain -> Nx[0], Nx2 = pDomain -> Nx[1], Nx3 = pDomain -> Nx[2];
  Real tmeanrho; //total rho

#ifdef MPI_PARALLEL
  int ierr;
  int myid = myID_Comm_world;
#else
  int myID = 0;
#endif

  /*Determine mean density on root Domain */
  if(pDomain -> Level == 0){
    tmeanrho = 0.0;
    for(k=ks;k<=ke;k++){
      for(j=js;j<=je;j++){
	for(i=is;i<=ie;i++){
	  tmeanrho += pG -> U[k][j][i].d;
	}
      }
    }

#ifdef MPI_PARALLEL
    ierr = MPI_Allreduce(&tmeanrho,&grav_mean_rho,1,MPI_DOUBLE,MPI_SUM,pDomain->Comm_Domain);
#else
    grav_mean_rho = tmeanrho;
#endif

    grav_mean_rho /= (Nx1*Nx2*Nx3);
  }

#ifdef MPI_PARALLEL
  MPI_Bcast(&grav_mean_rho,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
#endif
  return;
}
#endif /*SELF_GRAVITY*/


/*==============================================================================
 * PROBLEM PRIVATE USER HELPER FUNCTIONS
 * vector             - Creates array of size(1,n)
 * ivector            - Creates int array of size(1,n)
 * matrix             - Creates matrix of size(n1,n2) index starts at 1
 * free_vector        - Frees allocation of vector
 * free_ivector       - Frees allocation of ivector
 * free_matrix        - Frees allocation of matrix
 * nrerror            - Prints error message and closes application
 *-----------------------------------------------------------------------------*/
static double *vector(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
  double *v;

  v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
  if (!v) nrerror("allocation failure in dvector()");
  return v-nl+NR_END;
}

static int *ivector(long nl, long nh)
/* allocate an int vector with subscript range v[nl..nh] */
{
  int *v;

  v=(int *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(int)));
  if (!v) nrerror("allocation failure in dvector()");
  return v-nl+NR_END;
}
static double **matrix(long nrl, long nrh, long ncl, long nch)
/* allocate a double matrix with subscript range m[nrl. . nrh][ncl. . nch] */
{
  long i, nrow = nrh-nrl+1,ncol=nch-ncl+1;
  double **m;

  /*allocate pointers to rows */
  m=(double **) malloc((size_t)((nrow+NR_END)*sizeof(double*)));
  if (!m) nrerror("allocation failure 1 in matrix()");
  m += NR_END;
  m -= nrl;

  /* allocate rows and set pointers to them */
  m[nrl]=(double *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(double)));
  if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
  m[nrl] += NR_END;
  m[nrl] -= ncl;
  
  for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

  /* return pointer to array of pointers to rows */
  return m;
}

static void free_vector(double *v, long nl, long nh)
/* free a double vector allocated with vector() */
{
  free((FREE_ARG) (v+nl-NR_END));
}


static void free_ivector(int *v, long nl, long nh)
/* free an int vector allocated with ivector() */
{
  free((FREE_ARG) (v+nl-NR_END));
}

static void free_matrix(double **m, long nrl, long nrh, long ncl, long nch)
/* free a double matrix allocated by matrix() */
{
  free((FREE_ARG) (m[nrl]+ncl-NR_END));
  free((FREE_ARG) (m+nrl-NR_END));
}

static void nrerror(char error_text[])
/* Numerical Recipes standard error handler */
{
  printf("Numerical Recipes run-time error...\n");
  printf("%s\n",error_text);
  printf("...now exiting to system...\n");
  exit(1);
}
