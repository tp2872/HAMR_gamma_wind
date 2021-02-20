#include "decs_MPI.h"

void const_transport1_res(double(*restrict pb[NB_LOCAL])[NPR], int n){
	int i, j, z, k, ind0;

	#pragma omp parallel shared(n,n_ord,n_active,E_corn, F1, F2, F3, dx,pb, N1_GPU_offset,N2_GPU_offset,N3_GPU_offset, nthreads) private(i,j,z, ind0)
	{
		#pragma omp for collapse(3) schedule(static,(BS_1+2*D1)*(BS_2+2*D2)*(BS_3+2*D3)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n] * D1 - D1, (N1_GPU_offset[n] + BS_1)*D1, N2_GPU_offset[n] * D2 - D2, (N2_GPU_offset[n] + BS_2)*D2, N3_GPU_offset[n] * D3 - D3, (N3_GPU_offset[n] + BS_3)*D3){
			ind0 = index_3D(n, i, j, z);

			//calculate the corner values of the electric field by averaging the Godunov fluxes, see formula 7 balsara&spicer
			#if(N3G>0)
			E_corn[nl[n]][ind0][1] = 0.25*(F3[nl[n]][ind0][B2] + F3[nl[n]][index_3D(n, i, j - D2, z)][B2] - F2[nl[n]][ind0][B3] - F2[nl[n]][index_3D(n, i, j, z - D3)][B3]);
			E_corn[nl[n]][ind0][2] = 0.25*(F1[nl[n]][ind0][B3] + F1[nl[n]][index_3D(n, i, j, z - D3)][B3] - F3[nl[n]][ind0][B1] - F3[nl[n]][index_3D(n, i - D1, j, z)][B1]);
			#endif
			E_corn[nl[n]][ind0][3] = 0.25*(F2[nl[n]][ind0][B1] + F2[nl[n]][index_3D(n, i - D1, j, z)][B1] - F1[nl[n]][ind0][B2] - F1[nl[n]][index_3D(n, i, j - D2, z)][B2]);

			//upwind the electric field based on transverse gradients conform gardiner&stone 2005/2015, not yet tested
			#if(N3G>0)
			dE[nl[n]][ind0][LEFT][1][2] = (pb[nl[n]][ind0][E1] + F2[nl[n]][ind0][B3]);
			dE[nl[n]][ind0][LEFT][1][3] = (pb[nl[n]][ind0][E1] - F3[nl[n]][ind0][B2]);
			dE[nl[n]][ind0][LEFT][2][1] = (pb[nl[n]][ind0][E2] - F1[nl[n]][ind0][B3]);
			dE[nl[n]][ind0][LEFT][2][3] = (pb[nl[n]][ind0][E2] + F3[nl[n]][ind0][B1]);
			#endif
			dE[nl[n]][ind0][LEFT][3][1] = (pb[nl[n]][ind0][E3] + F1[nl[n]][ind0][B2]);
			dE[nl[n]][ind0][LEFT][3][2] = (pb[nl[n]][ind0][E3] - F2[nl[n]][ind0][B1]);

			#if(N3G>0)
			dE[nl[n]][ind0][RIGHT][1][2] = (-F2[nl[n]][index_3D(n, i, j + D2, z)][B3] - pb[nl[n]][ind0][E1]);
			dE[nl[n]][ind0][RIGHT][1][3] = (F3[nl[n]][index_3D(n, i, j, z + D3)][B2] - pb[nl[n]][ind0][E1]);
			dE[nl[n]][ind0][RIGHT][2][1] = (F1[nl[n]][index_3D(n, i + D1, j, z)][B3] - pb[nl[n]][ind0][E2]);
			dE[nl[n]][ind0][RIGHT][2][3] = (-F3[nl[n]][index_3D(n, i, j, z + D3)][B1] - pb[nl[n]][ind0][E2]);
			#endif
			dE[nl[n]][ind0][RIGHT][3][1] = (-F1[nl[n]][index_3D(n, i + D1, j, z)][B2] - pb[nl[n]][ind0][E3]);
			dE[nl[n]][ind0][RIGHT][3][2] = (F2[nl[n]][index_3D(n, i, j + D2, z)][B1] - pb[nl[n]][ind0][E3]);
		}

		#pragma omp for collapse(2) schedule(static,(BS_1+D1)*(BS_2+D2)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n] * D1, (N1_GPU_offset[n] + BS_1)*D1, N2_GPU_offset[n] * D2, (N2_GPU_offset[n] + BS_2)*D2, N3_GPU_offset[n] * D3, (N3_GPU_offset[n] + BS_3)*D3){
			ind0 = index_3D(n, i, j, z);

			E_corn[nl[n]][ind0][1] = 0.25*((-F2[nl[n]][ind0][B3] - (dE[nl[n]][ind0][LEFT][1][3] * (double)(F2[nl[n]][ind0][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i, j - D2, z)][LEFT][1][3] * (double)(F2[nl[n]][ind0][RHO]>0.0)))
				+ (-F2[nl[n]][index_3D(n, i, j, z - D3)][B3] + (dE[nl[n]][index_3D(n, i, j, z - D3)][RIGHT][1][3] * (double)(F2[nl[n]][index_3D(n, i, j, z - D3)][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i, j - D2, z - D3)][RIGHT][1][3] * (double)(F2[nl[n]][index_3D(n, i, j, z - D3)][RHO]>0.0)))
				+ (F3[nl[n]][ind0][B2] - (dE[nl[n]][ind0][LEFT][1][2] * (double)(F3[nl[n]][ind0][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i, j, z - D3)][LEFT][1][2] * (double)(F3[nl[n]][ind0][RHO]>0.0)))
				+ (F3[nl[n]][index_3D(n, i, j - D2, z)][B2] + (dE[nl[n]][index_3D(n, i, j - D2, z)][RIGHT][1][2] * (double)(F3[nl[n]][index_3D(n, i, j - D2, z)][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i, j - D2, z - D3)][RIGHT][1][2] * (double)(F3[nl[n]][index_3D(n, i, j - D2, z)][RHO]>0.0))));
			E_corn[nl[n]][ind0][2] = 0.25*((-F3[nl[n]][ind0][B1] - (dE[nl[n]][ind0][LEFT][2][1] * (double)(F3[nl[n]][ind0][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i, j, z - D3)][LEFT][2][1] * (double)(F3[nl[n]][ind0][RHO] > 0.0)))
				+ (-F3[nl[n]][index_3D(n, i - D1, j, z)][B1] + (dE[nl[n]][index_3D(n, i - D1, j, z)][RIGHT][2][1] * (double)(F3[nl[n]][index_3D(n, i - D1, j, z)][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i - D1, j, z - D3)][RIGHT][2][1] * (double)(F3[nl[n]][index_3D(n, i - D1, j, z)][RHO] > 0.0)))
				+ (F1[nl[n]][ind0][B3] - (dE[nl[n]][ind0][LEFT][2][3] * (double)(F1[nl[n]][ind0][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i - D1, j, z)][LEFT][2][3] * (double)(F1[nl[n]][ind0][RHO] > 0.0)))
				+ (F1[nl[n]][index_3D(n, i, j, z - D3)][B3] + (dE[nl[n]][index_3D(n, i, j, z - D3)][RIGHT][2][3] * (double)(F1[nl[n]][index_3D(n, i, j, z - D3)][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i - D1, j, z - D3)][RIGHT][2][3] * (double)(F1[nl[n]][index_3D(n, i, j, z - D3)][RHO] > 0.0))));
			E_corn[nl[n]][ind0][3] = 0.25*((F2[nl[n]][ind0][B1] - (dE[nl[n]][ind0][LEFT][3][1] * (double)(F2[nl[n]][ind0][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i, j - D2, z)][LEFT][3][1] * (double)(F2[nl[n]][ind0][RHO] > 0.0)))
				+ (F2[nl[n]][index_3D(n, i - D1, j, z)][B1] + (dE[nl[n]][index_3D(n, i - D1, j, z)][RIGHT][3][1] * (double)(F2[nl[n]][index_3D(n, i - D1, j, z)][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i - D1, j - D2, z)][RIGHT][3][1] * (double)(F2[nl[n]][index_3D(n, i - D1, j, z)][RHO] > 0.0)))
				+ (-F1[nl[n]][ind0][B2] - (dE[nl[n]][ind0][LEFT][3][2] * (double)(F1[nl[n]][ind0][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i - D1, j, z)][LEFT][3][2] * (double)(F1[nl[n]][ind0][RHO] > 0.0)))
				+ (-F1[nl[n]][index_3D(n, i, j - D2, z)][B2] + (dE[nl[n]][index_3D(n, i, j - D2, z)][RIGHT][3][2] * (double)(F1[nl[n]][index_3D(n, i, j - D2, z)][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i - D1, j - D2, z)][RIGHT][3][2] * (double)(F1[nl[n]][index_3D(n, i, j - D2, z)][RHO] > 0.0))));

			if (j == 0 || j == (int)(N2*pow((1 + REF_2), block[n][AMR_LEVEL2]))) E_corn[nl[n]][ind0][1] = 0.5*(-F2[nl[n]][ind0][B3] - F2[nl[n]][index_3D(n, i, j, z - D3)][B3]);
			if (j == 0 || j == (int)(N2*pow((1 + REF_2), block[n][AMR_LEVEL2]))) E_corn[nl[n]][ind0][3] = 0.0;
		}
	}
}

void const_transport1_M1_2_res(double(*restrict pb[NB_LOCAL])[NPR], int n) {
	int i, j, z, k, ind0;

	#pragma omp parallel shared(n,n_ord,n_active,E_corn, F1, F2, F3, dx,pb, N1_GPU_offset,N2_GPU_offset,N3_GPU_offset, nthreads) private(i,j,z, ind0)
	{
		#pragma omp for collapse(3) schedule(static,(BS_1+2*D1)*(BS_2+2*D2)*(BS_3+2*D3)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n] * D1 - D1, (N1_GPU_offset[n] + BS_1) * D1, N2_GPU_offset[n] * D2 - D2, (N2_GPU_offset[n] + BS_2) * D2, N3_GPU_offset[n] * D3 - D3, (N3_GPU_offset[n] + BS_3) * D3) {
			ind0 = index_3D(n, i, j, z);

			//calculate the corner values of the electric field by averaging the Godunov fluxes, see formula 7 balsara&spicer
			#if(N3G>0)
			E_corn[nl[n]][ind0][1] = 0.25 * (F3[nl[n]][ind0][B2] + F3[nl[n]][index_3D(n, i, j - D2, z)][B2] - F2[nl[n]][ind0][B3] - F2[nl[n]][index_3D(n, i, j, z - D3)][B3]);
			E_corn[nl[n]][ind0][2] = 0.25 * (F1[nl[n]][ind0][B3] + F1[nl[n]][index_3D(n, i, j, z - D3)][B3] - F3[nl[n]][ind0][B1] - F3[nl[n]][index_3D(n, i - D1, j, z)][B1]);
			#endif
			E_corn[nl[n]][ind0][3] = 0.25 * (F2[nl[n]][ind0][B1] + F2[nl[n]][index_3D(n, i - D1, j, z)][B1] - F1[nl[n]][ind0][B2] - F1[nl[n]][index_3D(n, i, j - D2, z)][B2]);

			//upwind the electric field based on transverse gradients conform gardiner&stone 2005/2015, not yet tested
			#if(N3G>0)
			dE[nl[n]][ind0][LEFT][1][2] = (pb[nl[n]][ind0][E1] + F2[nl[n]][ind0][B3]);
			dE[nl[n]][ind0][LEFT][1][3] = (pb[nl[n]][ind0][E1] - F3[nl[n]][ind0][B2]);
			dE[nl[n]][ind0][LEFT][2][1] = (pb[nl[n]][ind0][E2] - F1[nl[n]][ind0][B3]);
			dE[nl[n]][ind0][LEFT][2][3] = (pb[nl[n]][ind0][E2] + F3[nl[n]][ind0][B1]);
			#endif
			dE[nl[n]][ind0][LEFT][3][1] = (pb[nl[n]][ind0][E3] + F1[nl[n]][ind0][B2]);
			dE[nl[n]][ind0][LEFT][3][2] = (pb[nl[n]][ind0][E3] - F2[nl[n]][ind0][B1]);

			#if(N3G>0)
			dE[nl[n]][ind0][RIGHT][1][2] = (-F2[nl[n]][index_3D(n, i, j + D2, z)][B3] - pb[nl[n]][ind0][E1]);
			dE[nl[n]][ind0][RIGHT][1][3] = (F3[nl[n]][index_3D(n, i, j, z + D3)][B2] - pb[nl[n]][ind0][E1]);
			dE[nl[n]][ind0][RIGHT][2][1] = (F1[nl[n]][index_3D(n, i + D1, j, z)][B3] - pb[nl[n]][ind0][E2]);
			dE[nl[n]][ind0][RIGHT][2][3] = (-F3[nl[n]][index_3D(n, i, j, z + D3)][B1] - pb[nl[n]][ind0][E2]);
			#endif
			dE[nl[n]][ind0][RIGHT][3][1] = (-F1[nl[n]][index_3D(n, i + D1, j, z)][B2] - pb[nl[n]][ind0][E3]);
			dE[nl[n]][ind0][RIGHT][3][2] = (F2[nl[n]][index_3D(n, i, j + D2, z)][B1] - pb[nl[n]][ind0][E3]);
		}

		#pragma omp for collapse(2) schedule(static,(BS_1+D1)*(BS_2+D2)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n] * D1, (N1_GPU_offset[n] + BS_1) * D1, N2_GPU_offset[n] * D2, (N2_GPU_offset[n] + BS_2) * D2, N3_GPU_offset[n] * D3, (N3_GPU_offset[n] + BS_3) * D3) {
			ind0 = index_3D(n, i, j, z);
			E_corn[nl[n]][ind0][1] *= 0.5;
			E_corn[nl[n]][ind0][2] *= 0.5;
			E_corn[nl[n]][ind0][3] *= 0.5;

			E_corn[nl[n]][ind0][1] += 0.25 * 0.5 * ((-F2[nl[n]][ind0][B3] - (dE[nl[n]][ind0][LEFT][1][3] * (double)(F2[nl[n]][ind0][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i, j - D2, z)][LEFT][1][3] * (double)(F2[nl[n]][ind0][RHO] > 0.0)))
				+ (-F2[nl[n]][index_3D(n, i, j, z - D3)][B3] + (dE[nl[n]][index_3D(n, i, j, z - D3)][RIGHT][1][3] * (double)(F2[nl[n]][index_3D(n, i, j, z - D3)][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i, j - D2, z - D3)][RIGHT][1][3] * (double)(F2[nl[n]][index_3D(n, i, j, z - D3)][RHO] > 0.0)))
				+ (F3[nl[n]][ind0][B2] - (dE[nl[n]][ind0][LEFT][1][2] * (double)(F3[nl[n]][ind0][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i, j, z - D3)][LEFT][1][2] * (double)(F3[nl[n]][ind0][RHO] > 0.0)))
				+ (F3[nl[n]][index_3D(n, i, j - D2, z)][B2] + (dE[nl[n]][index_3D(n, i, j - D2, z)][RIGHT][1][2] * (double)(F3[nl[n]][index_3D(n, i, j - D2, z)][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i, j - D2, z - D3)][RIGHT][1][2] * (double)(F3[nl[n]][index_3D(n, i, j - D2, z)][RHO] > 0.0))));
			E_corn[nl[n]][ind0][2] += 0.25 * 0.5 * ((-F3[nl[n]][ind0][B1] - (dE[nl[n]][ind0][LEFT][2][1] * (double)(F3[nl[n]][ind0][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i, j, z - D3)][LEFT][2][1] * (double)(F3[nl[n]][ind0][RHO] > 0.0)))
				+ (-F3[nl[n]][index_3D(n, i - D1, j, z)][B1] + (dE[nl[n]][index_3D(n, i - D1, j, z)][RIGHT][2][1] * (double)(F3[nl[n]][index_3D(n, i - D1, j, z)][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i - D1, j, z - D3)][RIGHT][2][1] * (double)(F3[nl[n]][index_3D(n, i - D1, j, z)][RHO] > 0.0)))
				+ (F1[nl[n]][ind0][B3] - (dE[nl[n]][ind0][LEFT][2][3] * (double)(F1[nl[n]][ind0][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i - D1, j, z)][LEFT][2][3] * (double)(F1[nl[n]][ind0][RHO] > 0.0)))
				+ (F1[nl[n]][index_3D(n, i, j, z - D3)][B3] + (dE[nl[n]][index_3D(n, i, j, z - D3)][RIGHT][2][3] * (double)(F1[nl[n]][index_3D(n, i, j, z - D3)][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i - D1, j, z - D3)][RIGHT][2][3] * (double)(F1[nl[n]][index_3D(n, i, j, z - D3)][RHO] > 0.0))));
			E_corn[nl[n]][ind0][3] += 0.25 * 0.5 * ((F2[nl[n]][ind0][B1] - (dE[nl[n]][ind0][LEFT][3][1] * (double)(F2[nl[n]][ind0][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i, j - D2, z)][LEFT][3][1] * (double)(F2[nl[n]][ind0][RHO] > 0.0)))
				+ (F2[nl[n]][index_3D(n, i - D1, j, z)][B1] + (dE[nl[n]][index_3D(n, i - D1, j, z)][RIGHT][3][1] * (double)(F2[nl[n]][index_3D(n, i - D1, j, z)][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i - D1, j - D2, z)][RIGHT][3][1] * (double)(F2[nl[n]][index_3D(n, i - D1, j, z)][RHO] > 0.0)))
				+ (-F1[nl[n]][ind0][B2] - (dE[nl[n]][ind0][LEFT][3][2] * (double)(F1[nl[n]][ind0][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i - D1, j, z)][LEFT][3][2] * (double)(F1[nl[n]][ind0][RHO] > 0.0)))
				+ (-F1[nl[n]][index_3D(n, i, j - D2, z)][B2] + (dE[nl[n]][index_3D(n, i, j - D2, z)][RIGHT][3][2] * (double)(F1[nl[n]][index_3D(n, i, j - D2, z)][RHO] <= 0.0) + dE[nl[n]][index_3D(n, i - D1, j - D2, z)][RIGHT][3][2] * (double)(F1[nl[n]][index_3D(n, i, j - D2, z)][RHO] > 0.0))));

			#if(!CARTESIAN)
			if (j == 0 || j == (int)(N2 * pow((1 + REF_2), block[n][AMR_LEVEL2]))) E_corn[nl[n]][ind0][1] += 0.5 * 0.5 * (-F2[nl[n]][ind0][B3] - F2[nl[n]][index_3D(n, i, j, z - D3)][B3]);
			if (j == 0 || j == (int)(N2 * pow((1 + REF_2), block[n][AMR_LEVEL2]))) E_corn[nl[n]][ind0][3] = 0.0;
			#endif
		}
	}
}


void const_transport2_res(double(*restrict psi[NB_LOCAL])[NDIM], double(*restrict psf[NB_LOCAL])[NDIM], double Dt, int n){
	int i, j, z, k, ind0;
	#pragma omp parallel shared(n,n_ord,n_active,E_corn, gdet,psf,psi, dx,Dt, p, N1_GPU_offset,N2_GPU_offset,N3_GPU_offset, nthreads) private(i,j,z, ind0)
	{
		//update the staggered field components
		#pragma omp for collapse(3) schedule(static,(BS_1+D1)*(BS_2)*(BS_3)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + BS_1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1){
			ind0 = index_3D(n, i, j, z);
			psf[nl[n]][index_3D(n, i, j, z)][1] = psi[nl[n]][index_3D(n, i, j, z)][1] - Dt / dx[nl[n]][2] * (E_corn[nl[n]][index_3D(n, i, j + D2, z)][3] - E_corn[nl[n]][ind0][3]) / gdet[nl[n]][index_2D(n, i, j, z)][FACE1];
			#if(N3G>0)
			psf[nl[n]][index_3D(n, i, j, z)][1] += Dt / dx[nl[n]][3] * (E_corn[nl[n]][index_3D(n, i, j, z + D3)][2] - E_corn[nl[n]][ind0][2]) / gdet[nl[n]][index_2D(n, i, j, z)][FACE1];
			#endif
		}

		//update the staggered field components
		#pragma omp for collapse(3) schedule(static,(BS_1)*(BS_2+D2)*(BS_3)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + BS_1 - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2, N3_GPU_offset[n], N3_GPU_offset[n] + BS_3 - 1){
			ind0 = index_3D(n, i, j, z);
			psf[nl[n]][index_3D(n, i, j, z)][2] = psi[nl[n]][index_3D(n, i, j, z)][2] + Dt / dx[nl[n]][1] * (E_corn[nl[n]][index_3D(n, i + D1, j, z)][3] - E_corn[nl[n]][ind0][3]) / gdet[nl[n]][index_2D(n, i, j, z)][FACE2];
			#if(N3G>0)
			psf[nl[n]][index_3D(n, i, j, z)][2] -= Dt / dx[nl[n]][3] * (E_corn[nl[n]][index_3D(n, i, j, z + D3)][1] - E_corn[nl[n]][ind0][1]) / gdet[nl[n]][index_2D(n, i, j, z)][FACE2];
			#endif		
		}

		//update the staggered field components
		#if(N3G>0)
		#pragma omp for collapse(3) schedule(static,(BS_1)*(BS_2)*(BS_3+D3)/nthreads)
		ZSLOOP3D(N1_GPU_offset[n], N1_GPU_offset[n] + BS_1 - 1, N2_GPU_offset[n], N2_GPU_offset[n] + BS_2 - 1, N3_GPU_offset[n], (N3_GPU_offset[n] + BS_3)*D3){
			ind0 = index_3D(n, i, j, z);
			psf[nl[n]][index_3D(n, i, j, z)][3] = psi[nl[n]][index_3D(n, i, j, z)][3] = Dt / dx[nl[n]][1] * (E_corn[nl[n]][index_3D(n, i + D1, j, z)][2] - E_corn[nl[n]][ind0][2]) / gdet[nl[n]][index_2D(n, i, j, z)][FACE3]
				+ Dt / dx[nl[n]][2] * (E_corn[nl[n]][index_3D(n, i, j + D2, z)][1] - E_corn[nl[n]][ind0][1]) / gdet[nl[n]][index_2D(n, i, j, z)][FACE3];
		}
		#endif
	}
}
