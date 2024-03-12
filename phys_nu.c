#include "include.h"
#include "decs.h"

#if (NEUTRINOS_M1)
void ucon_calc_nu(double * restrict pr, struct of_geom * restrict geom, double * restrict ucon_nu, int species)
{
	double alpha, gamma_nu;
	double beta[NDIM];
	int j;

	alpha = 1. / sqrt(-geom->gcon[0][0]);
	#pragma ivdep
	SLOOPA beta[j] = geom->gcon[0][j] * alpha*alpha;

	if (gamma_calc_nu(pr, geom, &gamma_nu, species)) {
		fflush(stderr);
		fprintf(stderr, "\nucon_calc_nu(): gamma_nu failure \n");
		fflush(stderr);
		fail(FAIL_GAMMA);
	}

	ucon_nu[0] = gamma_nu / alpha;
	#pragma ivdep
	SLOOPA ucon_nu[j] = pr[index_nu(U1_NU, species) + j - 1] - gamma_nu*beta[j] / alpha;

	return;
}

int gamma_calc_nu(double* restrict pr, struct of_geom* restrict geom, double* restrict gamma_nu, int species)
{
	double qsq;
	qsq = geom->gcov[1][1] * pr[index_nu(U1_NU, species)] * pr[index_nu(U1_NU, species)] + geom->gcov[2][2] * pr[index_nu(U2_NU, species)] * pr[index_nu(U2_NU, species)] + geom->gcov[3][3] * pr[index_nu(U3_NU, species)] * pr[index_nu(U3_NU, species)] + 2. * (geom->gcov[1][2] * pr[index_nu(U1_NU, species)] * pr[index_nu(U2_NU, species)] + geom->gcov[1][3] * pr[index_nu(U1_NU, species)] * pr[index_nu(U3_NU, species)] + geom->gcov[2][3] * pr[index_nu(U2_NU, species)] * pr[index_nu(U3_NU, species)]);
	if (qsq < 0.) {
		if (fabs(qsq) > 1.E-10) { // then assume not just machine precision
			fprintf(stderr, "gamma_calc_nu():  failed: qsq = %28.18e \n", qsq);
			fprintf(stderr, "v[1-3] = %28.18e %28.18e %28.18e  \n", pr[index_nu(U1_NU, species)], pr[index_nu(U2_NU, species)], pr[index_nu(U3_NU, species)]);
			*gamma_nu = 1.;
			return (1);
		}
		else qsq = 1.E-10; // set floor
	}

	*gamma_nu = sqrt(1. + qsq);

	return(0);
}
#endif