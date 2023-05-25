#include "include.h"
#include "decs_MPI.h"

/** 
 *
 * this file contains all the coordinate dependent
 * parts of the code, except the initial and boundary
 * conditions 
 *
 **/
/***************************************************************************/
/***************************************************************************
coord():
-------
-- given the indices i,j and location in the cell, return with
the values of X1,X2 there;
-- the locations are defined by :
-----------------------
|                     |
|                     |
|FACE1   CENT         |
|                     |
|CORN    FACE2        |
----------------------
***************************************************************************/
void coord(int n, int i, int j, int z, int loc, double * restrict X)
{
	X[0] = 0.0;
	int j_local = j;
	#if(SPHERICAL || SPHERICAL_GR)
	if (j < 0) j_local = -j - 1;
	if (j >= N2*pow(1 + REF_2, block[n][AMR_LEVEL2])) j_local = 2 * N2*pow(1 + REF_2, block[n][AMR_LEVEL2]) - 1 - j;
	if (j == N2*pow(1 + REF_2, block[n][AMR_LEVEL2]) && loc == FACE2) j_local = j;
	#endif
	if (loc == FACE1) {
		X[1] = startx[1] + i*dx[nl[n]][1];
		X[2] = startx[2] + (j_local + 0.5)*dx[nl[n]][2];
		X[3] = startx[3] + (z + 0.5)*dx[nl[n]][3];
	}
	else if (loc == FACE2) {
		X[1] = startx[1] + (i + 0.5)*dx[nl[n]][1];
		X[2] = startx[2] + j_local*dx[nl[n]][2];
		X[3] = startx[3] + (z + 0.5)*dx[nl[n]][3];
	}
	else if (loc == FACE3) {
		X[1] = startx[1] + (i + 0.5)*dx[nl[n]][1];
		X[2] = startx[2] + (j_local + 0.5)*dx[nl[n]][2];
		X[3] = startx[3] + z*dx[nl[n]][3];
	}
	else if (loc == CENT) {
		X[1] = startx[1] + (i + 0.5)*dx[nl[n]][1];
		X[2] = startx[2] + (j_local + 0.5)*dx[nl[n]][2];
		X[3] = startx[3] + (z + 0.5)*dx[nl[n]][3];
	}
	else {
		X[1] = startx[1] + i*dx[nl[n]][1];
		X[2] = startx[2] + j_local*dx[nl[n]][2];
		X[3] = startx[3] + z*dx[nl[n]][3];
	}

	#if(SPHERICAL || SPHERICAL_GR)
	if (j < 0){
		X[2] = X[2] + 1;
		X[2] = -X[2];
		X[2] = X[2] - 1;
	}
	if (j == N2*pow(1 + REF_2, block[n][AMR_LEVEL2]) && loc == FACE2){
	}
	else if (j >= N2*pow(1 + REF_2, block[n][AMR_LEVEL2])){
		X[2] = X[2] + 1;
		X[2] = 4. - X[2];
		X[2] = X[2] - 1;
	}
	#endif

	return;
}

/* should return boyer-lindquist coordinte of point */
void bl_coord(double * restrict X, double * restrict r, double * restrict th, double * restrict phi)
{
	double V[4];
  void (*vofx_function_pointer)(double*, double*);

  //choose the type of coordinates depending on the problem at hand
  #if( WHICHPROBLEM == POSTMERGER_PROBLEM)
	vofx_function_pointer = vofx_matthewcoords; // vofx_sjetcoords;
  #else
    vofx_function_pointer = vofx_matthewcoords;
  #endif
  
	#if(!DOCYLINDRIFYCOORDS)
    vofx_function_pointer(X,V);
	#else
    vofx_cylindrified(X, vofx_function_pointer, V);
	#endif

	// avoid singularity at polar axis
	#if(COORDSINGFIX)
	if (fabs(V[2])<SINGSMALL){
		if (V[2] >= 0.0) V[2] = SINGSMALL;
		if (V[2]<0.0)  V[2] = -SINGSMALL;
	}
	if (fabs(M_PI - V[2]) <SINGSMALL){
		if (V[2] >= M_PI) V[2] = M_PI + SINGSMALL;
		if (V[2]<M_PI)  V[2] = M_PI -  SINGSMALL;
	}
	#endif

	*r = V[1];
	*th = V[2];
	*phi = V[3];
	return ;
}

void vofx_matthewcoords(double *X, double *V){
	#if(CARTESIAN || CARTESIAN_GR)
	double r, R, x, y, z;
	x = X[1];
	y = X[2];
	z = X[3];
	//if (fabs(x) < 0.0001) x = 0.0001;
	//if (fabs(y) < 0.0001) y = 0.0001;
	if (fabs(z) < 0.000001) z = 0.000001;

	R = sqrt(x * x + y * y + z * z);
	//R = MY_MAX(R, 1.0);
	r =  sqrt(0.5 * (R * R - a * a + sqrt(pow(R * R - a * a, 2.0) + 4.0 * a * a * z * z)));
	if (!isfinite(r))fprintf(stderr, "Error 1: R not finite! \n");

	//r = R;
	V[1] = r;
	V[2] = acos(z / r);
	if (!isfinite(V[2]))fprintf(stderr, "Error 2: V2 not finite! %f %f \n", r,z);
	V[3] = atan2(y, x) + atan(a / r);
	if (!isfinite(V[3]))fprintf(stderr, "Error 3: V3 not finite! %f %f %f \n", x / (sqrt(r * r + a * a) * sin(V[2])), atan(a/r));

	#else
	V[0] = X[0];
	double Xtrans = pow(log(RTRANS - RB), 1. / RADEXP);
	if (X[1] < Xtrans){
		V[1] = exp(pow(X[1], RADEXP)) + RB;
	}
	else if (X[1] >= Xtrans && X[1]<1.01*Xtrans){
		V[1] = 10.*(X[1] / Xtrans - 1.)*((X[1] - Xtrans)*RADEXP*exp(pow(Xtrans, RADEXP))*pow(Xtrans, -1. + RADEXP) + RTRANS) +
			(1. - 10.*(X[1] / Xtrans - 1.))*(exp(pow(X[1], RADEXP)) + RB);
	}
	else{
		V[1] = (X[1] - Xtrans)*RADEXP*exp(pow(Xtrans, RADEXP))*pow(Xtrans, -1. + RADEXP) + RTRANS;
	}
	double A1 = 1. / (1. + pow(CHARLIE*(log(V[1]) / log(10.)), DELTA));
	double A2 = BRAVO*(log(V[1]) / log(10.)) + TANGO;
	double A3 = pow(0.5, 1. - A2);
	double sign = 1.;
	double X_2 =(X[2]+1.0)/2.0;
	double Xc = sqrt(pow(X_2, 2.));

	if (X_2 < 0.0){
		sign = -1.;
	}
	if (X_2 > 1.0){
		sign = -1.;
		Xc = 2. - Xc;
	}
	if (X_2 >= 0.5){
		Xc = 1. - Xc;
		V[2] = M_PI - sign*(A1* M_PI*Xc + M_PI*(1. - A1)*(A3*pow(Xc, A2) + 0.50 / M_PI*sin(M_PI + 2.*M_PI*(A3*pow(Xc, A2)))));
	}
	else{
		V[2] = sign*(A1* M_PI*Xc + M_PI*(1. - A1)*(A3*pow(Xc, A2) + 0.50 / M_PI*sin(M_PI + 2.*M_PI*(A3*pow(Xc, A2)))));
	}
	V[3] = X[3];
	#endif
}

void vofx_sjetcoords( double *X, double *V )
{
  double thetaofx2(double x2, double ror0nu);

  /////////////////////
  //ANGULAR GRID SETUP
  /////////////////////
  
  //transverse resolution fraction devoted to different components
  //(sum should be <1)
  double global_fracdisk = 0.36;
  double global_fracjet = 0.15;
  
  double global_jetnu1 = -2.;  //the nu-parameter that determines jet shape
  double global_jetnu2 = 0.75;  //the nu-parameter that determines jet shape
  
  //subtractor, controls the size of the last few cells close to axis:
  //if rsjet = 0, then no modification <- *** default for use with grid cylindrification
  //if rsjet ~ 0.5, the grid is nearly vertical rather than monopolar,
  //                which makes the timestep larger
  double global_rsjet = 0.0;
  
  //distance at which theta-resolution is *exactly* uniform in the jet grid -- want to have this at BH horizon;
  //otherwise, near-uniform near jet axis but less resolution (much) further from it
  //the larger r0grid, the larger the thickness of the jet
  //to resolve
  double global_r0grid = Rin;
  
  //distance at which jet part of the grid becomes monopolar
  //should be the same as r0disk to avoid cell crowding at the interface of jet and disk grids
  double global_r0jet = 40*Rin;
  
  //distance after which the jet grid collimates according to the usual jet formula
  //the larger this distance, the wider is the jet region of the grid
  double global_rjetend = 1e3;
  
  //distance at which disk part of the grid becomes monopolar
  //the larger r0disk, the larger the thickness of the disk
  //to resolve
  double global_r0disk = 2*Rin;
  
  //distance after which the disk grid collimates to merge with the jet grid
  //should be roughly outer edge of the disk
  double global_rdiskend = 1.e7;

  
  //for SJETCOORDS
  double theexp;
  double Ftrgen( double x, double xa, double xb, double ya, double yb );
  double limlin( double x, double x0, double dx, double y0 );
  double minlin( double x, double x0, double dx, double y0 );
  double mins( double f1, double f2, double df );
  double maxs( double f1, double f2, double df );
  double thetaofx2(double x2, double ror0nu);
  double  fac, faker, ror0nu;
  double fakerdisk, fakerjet;
  double rbeforedisk, rinsidedisk, rinsidediskmax, rafterdisk;
  double ror0nudisk, ror0nujet, thetadisk, thetajet;
  
  V[0] = X[0];
  
  theexp = X[1];
  
  if( X[1] > x1br ) {
    theexp += cpow2 * pow(X[1]-x1br,npow2);
  }
  V[1] = R0+exp(theexp);
  
  double r1disk, r1jet, r2jet, r1, dr;
  fac = Ftrgen( fabs(X[2]), global_fracdisk, 1-global_fracjet, 0, 1 );
  
  r1disk = mins( V[1]/global_r0disk, 1. , 0.5 ) * (global_r0disk/global_r0grid);
  //r2disk = V[1]/r1;
  
  if( global_r0disk >= global_r0jet ) {
    r1jet = mins( V[1]/global_r0jet, 1. , 0.5 ) * (global_r0jet/global_r0grid);
    r2jet = V[1]/(r1jet*global_r0grid);
    dr = global_rjetend/global_r0jet;
    r2jet = mins( r2jet, dr, 0.5*dr );
  }
  else {
    r1jet = mins( V[1]/global_r0disk, 1. , 0.5 ) * (global_r0disk/global_r0grid);
    r2jet = maxs( V[1]/global_r0jet, 1., 0.5);
    dr = global_rjetend/global_r0jet;
    r2jet = mins( r2jet, dr, 0.5*dr );
  }
  
  ror0nudisk = pow( r1disk, 0.5*global_jetnu1);
  ror0nujet = pow( r1jet, 0.5*global_jetnu1) * pow(r2jet, 0.5*global_jetnu2);
  
  thetadisk = thetaofx2( X[2], ror0nudisk );
  thetajet = thetaofx2( X[2], ror0nujet );
  V[2] = fac*thetajet + (1 - fac)*thetadisk;
  
  // default is uniform \phi grid
  V[3]=X[3];
}

double thetaofx2(double x2, double ror0nu)
{
  double theta;
  if( x2 < -0.5 ) {
    theta = 0       + atan( tan((x2+1)*M_PI_2)/ror0nu );
  }
  else if( x2 >  0.5 ) {
    theta = M_PI    + atan( tan((x2-1)*M_PI_2)/ror0nu );
  }
  else {
    theta = M_PI_2 + atan( tan(x2*M_PI_2)*ror0nu );
  }
  return(theta);
}

/* some grid location, dxs */
void set_points(int n)
{
	#if(CARTESIAN || CARTESIAN_GR)
		#if(WHICHPROBLEM==SHOCK_TUBE)
		dx[nl[n]][1] = 2.2 / (double)(N1) / (double)(pow(1 + REF_1, block[n][AMR_LEVEL1]));
		dx[nl[n]][2] = 2.2 / (double)(N2) / (double)(pow(1 + REF_2, block[n][AMR_LEVEL2]));
		dx[nl[n]][3] = 2.2 / (double)(N3) / (double)(pow(1 + REF_3, block[n][AMR_LEVEL3]));
		#elif(WHICHPROBLEM==SHOCK_TUBE)
		dx[nl[n]][1] = 100. / (double)(N1) / (double)(pow(1 + REF_1, block[n][AMR_LEVEL1]));
		dx[nl[n]][2] = 100. / (double)(N2) / (double)(pow(1 + REF_2, block[n][AMR_LEVEL2]));
		dx[nl[n]][3] = 100. / (double)(N3) / (double)(pow(1 + REF_3, block[n][AMR_LEVEL3]));
		#else
		dx[nl[n]][1] = 2 * Rout / (double)(N1) / (double)(pow(1 + REF_1, block[n][AMR_LEVEL1]));
		dx[nl[n]][2] = 2 * Rout / (double)(N2) / (double)(pow(1 + REF_2, block[n][AMR_LEVEL2]));
		dx[nl[n]][3] = 2 * Rout / (double)(N3) / (double)(pow(1 + REF_3, block[n][AMR_LEVEL3]));
		#endif
	#else
	double Xtrans = pow(log(RTRANS - RB), 1. / RADEXP);
	if(Rout<=RTRANS){
		dx[nl[n]][1] = (pow(log(Rout - RB), 1. / RADEXP) - pow(log(Rin - RB), 1. / RADEXP)) / (double)(N1) / (double)(pow(1 + REF_1, block[n][AMR_LEVEL1]));
	}
	else{
		dx[nl[n]][1] = ((Rout - RTRANS + Xtrans *RADEXP*exp(pow(Xtrans, RADEXP))*pow(Xtrans, -1. + RADEXP)) / (RADEXP*exp(pow(Xtrans, RADEXP))*
			pow(Xtrans, -1. + RADEXP)) - pow(log(Rin), 1. / RADEXP)) / (double)(N1) / (double)(pow(1 + REF_1, block[n][AMR_LEVEL1]));
	}
	dx[nl[n]][2] = 2.*fractheta / (double)(N2) / (double)(pow(1 + REF_2, block[n][AMR_LEVEL2]));
	dx[nl[n]][3] = 2.*M_PI / (double)(N3) / (double)(pow(1 + REF_3, block[n][AMR_LEVEL3]));
	#endif
}

void set_gridparam(void) {
	a = BH_SPIN;
	Rin = 0.85 * (1. + sqrt(1. - a * a)); 	
	Rout = ROUT;
	lim = MC;
	failed = 0;
	cour = COUR;
	if (dt > 1e-5) dt = dt;
	else dt = 1.e-4;
	R0 = 0.0;
	gam = GAMMA;

	#if(CARTESIAN || CARTESIAN_GR)
		//Leave Rin to this value for CKS coordinates
		Rin = 0.5 * (1. + sqrt(1. - a * a));
		#if(WHICHPROBLEM==SHOCK_TUBE)
		startx[1] = -1.1;
		startx[2] = -1.1;
		startx[3] = -1.1;
		#elif(WHICHPROBLEM==RAD_PULSE)
		startx[1] = -50.;
		startx[2] = -50.;
		startx[3] = -50.;
		#else
		startx[1] = -Rout;
		startx[2] = -Rout;
		startx[3] = -Rout;
		#endif
	#else
	if (N2 != 1) {
		//2D problem, use full pi-wedge in theta
		#if(TRANS_BOUND_SMALL)
		fractheta = 1.0 - 1.0e-13;
		#else
		fractheta = 1.0 - 2.0 / ((double)N2)*(BOUND_TYPE2 == TRANSMISSIVE);
		#endif
	}
	else {
		//1D problem (since only 1 cell in theta-direction), use a restricted theta-wedge
		fractheta = 1.e-2;
	}
	//#if(WHICHPROBLEM == POSTMERGER_PROBLEM)
	#if (0)
	const double RELACC = 1e-14;
	const int ITERMAX = 50;
	rbr = 1e+4;
	npow2=4.0; //power exponent
	cpow2=1.0; //exponent prefactor (the larger it is, the more hyperexponentiation is)
	double x1max0, dxmax;
	int iter;
  
	Rin = 0.98 * (1. + sqrt(1. - a * a));  //.98
	Rout = 1e6;
	x1br = log( rbr - R0 );
  
	if( Rout < rbr ) {
	x1max = log(Rout-R0);
	}
	else {
	x1max0 = 1.;
	x1max = 2.;
    
	//find the root via iterations
	for( iter = 0; iter < ITERMAX; iter++ ) {
		if( fabs((x1max - x1max0)/x1max) < RELACC ) {
		break;
		}
		x1max0 = x1max;
		dxmax= (pow( (log(Rout-R0) - x1max0)/cpow2, 1./npow2 ) + x1br) - x1max0;
      
		// need a slight damping factor
		double dampingfactor=0.5;
		x1max = x1max0 + dampingfactor*dxmax;
		if (x1max> log(Rout-R0)){x1max = log(Rout-R0);}
	}
    
	if( iter == ITERMAX ) {
		if(rank==0) {
		printf( "Error: iteration procedure for finding x1max has not converged: x1max = %g, dx1max/x1max = %g, iter = %d\n",
				x1max, (x1max-x1max0)/x1max, iter );
		printf( "Error: iteration procedure for finding x1max has not converged: rbr= %g, x1br = %g, log(Rout-R0) = %g\n",
				rbr, x1br, log(Rout-R0) );
		}
		exit(1);
	}
	else {
		if(rank==0) printf( "x1max = %g (dx1max/x1max = %g, itno = %d)\n", x1max, (x1max-x1max0)/x1max, iter );
	}
	}
	startx[1] = log(Rin - R0) ;  //minimum values
	startx[2] = -1.+(1.-fractheta) ;   //minimum values
	startx[3] = 0. ;   //minimum values
	#else
	startx[1] = pow(log(Rin - RB), 1. / RADEXP);
	startx[2] = -1. + 1.*(1. - fractheta);
	startx[3] = 0.;
	#endif
	#endif
}

//////////////////////////////////////////////////////////////////////////////////////////
//
//  CYLINDRIFICATION
//
//////////////////////////////////////////////////////////////////////////////////////////
//Adjusts V[2]=theta so that a few innermost cells around the pole
//become cylindrical
//ASSUMES: poles are at
//            X[2] = -1 and +1, which correspond to
//            V[2] = 0 and pi
void vofx_cylindrified(double *Xin, void(*vofx)(double*, double*), double *Vout)
{
	double npiovertwos;
	double X[NDIM], V[NDIM];
	double Vin[NDIM];
	double X0[NDIM], V0[NDIM];
	double Xtr[NDIM], Vtr[NDIM];
	double f1, f2, dftr;
	double sinth, th;
	int j, ismirrored;

	vofx(Xin, Vin);

	// BRING INPUT TO 1ST QUADRANT:  X[2] \in [-1 and 0]
	to1stquadrant(Xin, X, &ismirrored);
	vofx(X, V);

	//initialize X0: cylindrify region
	//X[1] < X0[1] && X[2] < X0[2] (value of X0[3] not used)
	X0[0] = Xin[0];
  
  //{0, roughly midpoint between grid origin and x10, -1, 0}
  DLOOPA Xtr[j] = X[j];

#if( WHICHPROBLEM == POSTMERGER_PROBLEM)
	X0[1] = 3.0;
	X0[2] = -1. + 1./256.;
	X0[3] = 0.;
  Xtr[1] = log( 0.5*( exp(X0[1])+exp(startx[1]) ) );   //always bound to be between startx[1] and X0[1]
#else
  /*disk 150^3 Rout 100 Rg-->100^3=25 Rg*/
  X0[1] = pow(log(38.*(double)N3 / 250.0 - RB), 1. / RADEXP);
  X0[2] = -1. + 1. / ((double)(N2));
  X0[3] = 0.;
  //3D jet
  //Xtr[1] = pow(log(0.5*(exp(pow(X0[1], RADEXP) + RB) + exp(pow(startx[1], RADEXP) + RB))), 1. / RADEXP);   //always bound to be between startx[1] and X0[1]
  Xtr[1] = pow(log(0.5*(exp(pow(X0[1],RADEXP))+RB + exp(pow(startx[1],RADEXP))+RB)-RB),1./RADEXP);   //always bound to be between startx[1] and X0[1]
#endif
	/*3D jet Rout 10000 Rg 1024x400x100*/
	/*X0[1] = pow(log(600. - RB), 1. / RADEXP);
	X0[2] = -1. + 3. / (double)N2;
	X0[3] = 0.;*/
	vofx(X0, V0);

    //{0, roughly midpoint between grid origin and x10, -1, 0}
    DLOOPA Xtr[j] = X[j];
    //3D jet
    //Xtr[1] = pow(log(0.5*(exp(pow(X0[1], RADEXP) + RB) + exp(pow(startx[1], RADEXP) + RB))), 1. / RADEXP);   //always bound to be between startx[1] and X0[1]
    Xtr[1] = pow(log(0.5*(exp(pow(X0[1],RADEXP))+RB + exp(pow(startx[1],RADEXP))+RB)-RB),1./RADEXP);   //always bound to be between startx[1] and X0[1]
	vofx(Xtr, Vtr);

	f1 = func1(X0, X, vofx);
	f2 = func2(X0, X, vofx);
	dftr = func2(X0, Xtr, vofx) - func1(X0, Xtr, vofx);

	// Compute new theta
	sinth = maxs(V[1] * f1, V[1] * f2, Vtr[1] * fabs(dftr) + SMALL) / V[1];

	th = asin(sinth);

	//initialize Vout with the original values
	DLOOPA Vout[j] = Vin[j];

	//apply change in theta in the original quadrant
	if (0 == ismirrored) {
		Vout[2] = Vin[2] + (th - V[2]);
	}
	else {
		//if mirrrored, flip the sign
		Vout[2] = Vin[2] - (th - V[2]);
	}
}
//smooth step function:
// Ftr = 0 if x < 0, Ftr = 1 if x > 1 and smoothly interps. in btw.
double Ftr(double x)
{
	double res;

	if (x <= 0.) {
		res = 0.;
	}
	else if (x >= 1) {
		res = 1.;
	}
	else {
		res = (64. + cos(5. * M_PI*x) + 70. * sin((M_PI*(-1. + 2. * x)) / 2.) + 5. * sin((3. * M_PI*(-1. + 2. * x)) / 2.)) / 128.;
	}

	return(res);
}

double Ftrgenlin(double x, double xa, double xb, double ya, double yb)
{
	double Ftr(double x);
	double res;

	res = (x*ya) / xa + (-((x*ya) / xa) + ((x - xb)*(1. - yb)) / (1. - xb) + yb)*Ftr((x - xa) / (-xa + xb));

	return(res);
}

//goes from ya to yb as x goes from xa to xb
double Ftrgen(double x, double xa, double xb, double ya, double yb)
{
	double Ftr(double x);
	double res;

	res = ya + (yb - ya)*Ftr((x - xa) / (xb - xa));

	return(res);
}

double Fangle(double x)
{
	double res;

	if (x <= -1.) {
		res = 0.;
	}
	else if (x >= 1.) {
		res = x;
	}
	else {
		res = (1. + x + (-140. * sin((M_PI*(1. + x)) / 2.) + (10. * sin((3. * M_PI*(1. + x)) / 2.)) / 3. + (2. * sin((5. * M_PI*(1. + x)) / 2.)) / 5.) / (64.*M_PI)) / 2.;
	}

	return(res);

}

double limlin(double x, double x0, double dx, double y0)
{
	double Fangle(double x);
	return(y0 - dx * Fangle(-(x - x0) / dx));
}

double minlin(double x, double x0, double dx, double y0)
{
	double Fangle(double x);
	return(y0 + dx * Fangle((x - x0) / dx));
}

double mins(double f1, double f2, double df)
{
	double limlin(double x, double x0, double dx, double y0);
	return(limlin(f1, f2, df, f2));
}

double maxs(double f1, double f2, double df)
{
	double mins(double f1, double f2, double df);
	return(-mins(-f1, -f2, df));
}

//=mins if dir < 0
//=maxs if dir >= 0
double minmaxs(double f1, double f2, double df, double dir)
{
	double mins(double f1, double f2, double df);
	double maxs(double f1, double f2, double df);
	if (dir >= 0) {
		return(maxs(f1, f2, df));
	}

	return(mins(f1, f2, df));
}

//Converts copies Xin to Xout and converts
//but sets Xout[2] to lie in the 1st quadrant, i.e. Xout[2] \in [-1,0])
//if the point had to be mirrored
void to1stquadrant(double *Xin, double *Xout, int *ismirrored)
{
	double ntimes;
	int j;

	DLOOPA Xout[j] = Xin[j];

	//bring the angle variables to -2..2 (for X) and -2pi..2pi (for V)
	ntimes = floor((Xin[2] + 2.0) / 4.0);
	//this forces -2 < Xout[2] < 2
	Xout[2] -= 4. * ntimes;

	*ismirrored = 0;

	if (Xout[2] > 0.) {
		Xout[2] = -Xout[2];
		*ismirrored = 1 - *ismirrored;
	}

	//now force -1 < Xout[2] < 0
	if (Xout[2] < -1.) {
		Xout[2] = -2. - Xout[2];
		*ismirrored = 1 - *ismirrored;
	}
}

double sinth0(double *X0, double *X, void(*vofx)(double*, double*))
{
	double V0[NDIM];
	double Vc0[NDIM];
	double Xc0[NDIM];
	int j;

	//X1 = {0, X[1], X0[1], 0}
	DLOOPA Xc0[j] = X[j];
	Xc0[2] = X0[2];

	vofx(Xc0, Vc0);
	vofx(X0, V0);


	return(V0[1] * sin(V0[2]) / Vc0[1]);
}

double sinth1in(double *X0, double *X, void(*vofx)(double*, double*))
{
	double V[NDIM];
	double V0[NDIM];
	double V0c[NDIM];
	double X0c[NDIM];
	int j;

	//X1 = {0, X[1], X0[1], 0}
	DLOOPA X0c[j] = X0[j];
	X0c[2] = X[2];

	vofx(X, V);
	vofx(X0c, V0c);
	vofx(X0, V0);

	return(V0[1] * sin(V0c[2]) / V[1]);
}


double th2in(double *X0, double *X, void(*vofx)(double*, double*))
{
	double V[NDIM];
	double V0[NDIM];
	double Vc0[NDIM];
	double Xc0[NDIM];
	double Xcmid[NDIM];
	double Vcmid[NDIM];
	int j;
	double res;
	double th0;

	DLOOPA Xc0[j] = X[j];
	Xc0[2] = X0[2];
	vofx(Xc0, Vc0);

	DLOOPA Xcmid[j] = X[j];
	Xcmid[2] = 0.;
	vofx(Xcmid, Vcmid);

	vofx(X0, V0);
	vofx(X, V);

	th0 = asin(sinth0(X0, X, vofx));

	res = (V[2] - Vc0[2]) / (Vcmid[2] - Vc0[2]) * (Vcmid[2] - th0) + th0;

	return(res);
}

double func1(double *X0, double *X, void(*vofx)(double*, double*))
{
	double V[NDIM];

	vofx(X, V);

	return(sin(V[2]));
}

double func2(double *X0, double *X, void(*vofx)(double*, double*))
{
	double V[NDIM];
	double Xca[NDIM];
	double func2;
	int j;
	double sth1in, sth2in, sth1inaxis, sth2inaxis;

	//{0, X[1], -1, 0}
	DLOOPA Xca[j] = X[j];
	Xca[2] = -1.;

	vofx(X, V);

	sth1in = sinth1in(X0, X, vofx);
	sth2in = sin(th2in(X0, X, vofx));

	sth1inaxis = sinth1in(X0, Xca, vofx);
	sth2inaxis = sin(th2in(X0, Xca, vofx));

	func2 = minmaxs(sth1in, sth2in, fabs(sth2inaxis - sth1inaxis) + SMALL, X[1] - X0[1]);

	return(func2);
}
