
//4D Matrix Inversion
__device__ int invert_matrix_4D(double Am[][NDIM], double Aminv[][NDIM]){
	int i, j;
	int permute[NDIM];
	double dxm[NDIM], Amtmp[NDIM][NDIM];

	for (i = 0; i < NDIM*NDIM; i++) Amtmp[0][i] = Am[0][i];

	//Get the LU matrix:
	if (LU_decompose(Amtmp, permute) != 0) return(1);

	for (i = 0; i < NDIM; i++) {
		for (j = 0; j < NDIM; j++) { dxm[j] = 0.; }
		dxm[i] = 1.;

		//Solve the linear system for the i^th column of the inverse matrix
		LU_substitution(Amtmp, dxm, permute);

		for (j = 0; j < NDIM; j++) Aminv[j][i] = dxm[j];
	}

	return(0);
}

//2D Matrix inversion
__device__ int invert_matrix_1D(double Am[][1], double Aminv[][1])
{
	double D;

	D = 1.0 / (Am[0][0]);
	if (!isfinite(D)) {
		Aminv[0][0] = D;
		return(0);
	}
	else {
		Aminv[0][0] = D;
		return(0);
	}
}


//2D Matrix inversion
__device__ int invert_matrix_2D(double Am[][2], double Aminv[][2])
{
	double D, temp[2][2];

	D = 1.0 / (Am[0][0] * Am[1][1] - Am[1][0] * Am[0][1]);
	if (!isfinite(D)) {
		return(1);
	}
	else {
		temp[0][0] = D * Am[1][1];
		temp[0][1] = -D * Am[0][1];
		temp[1][0] = -D * Am[1][0];
		temp[1][1] = D * Am[0][0];

		Aminv[0][0] = temp[0][0];
		Aminv[1][0] = temp[1][0];
		Aminv[0][1] = temp[0][1];
		Aminv[1][1] = temp[1][1];

		return(0);
	}
}

//3D Matrix inversion
__device__ int invert_matrix_3D(double Am[][3], double Aminv[][3])
{

	int i, j;
	int n = 3;
	int permute[3];
	double dxm[3], Amtmp[3][3];

	for (i = 0; i < 3 * 3; i++) { Amtmp[0][i] = Am[0][i]; }

	// Get the LU matrix:
	if (LU_decompose_3D(Amtmp, permute) != 0) {
		return(1);
	}

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) { dxm[j] = 0.; }
		dxm[i] = 1.;

		/* Solve the linear system for the i^th column of the inverse matrix: :  */
		LU_substitution_3D(Amtmp, dxm, permute);

		for (j = 0; j < n; j++) { Aminv[j][i] = dxm[j]; }

	}

	return(0);
}

//5D Matrix inversion
__device__ int invert_matrix_5D(double Am[][5], double Aminv[][5])
{

	int i, j;
	int n = 5;
	int permute[5];
	double dxm[5], Amtmp[5][5];

	for (i = 0; i < 5 * 5; i++) { Amtmp[0][i] = Am[0][i]; }

	// Get the LU matrix:
	if (LU_decompose_5D(Amtmp, permute) != 0) {
		return(1);
	}

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) { dxm[j] = 0.; }
		dxm[i] = 1.;

		/* Solve the linear system for the i^th column of the inverse matrix: :  */
		LU_substitution_5D(Amtmp, dxm, permute);

		for (j = 0; j < n; j++) { Aminv[j][i] = dxm[j]; }

	}

	return(0);
}

//6D Matrix inversion
__device__ int invert_matrix_6D(double Am[][6], double Aminv[][6])
{

	int i, j;
	int n = 6;
	int permute[6];
	double dxm[6], Amtmp[6][6];

	for (i = 0; i < 6 * 6; i++) { Amtmp[0][i] = Am[0][i]; }

	// Get the LU matrix:
	if (LU_decompose_6D(Amtmp, permute) != 0) {
		return(1);
	}

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) { dxm[j] = 0.; }
		dxm[i] = 1.;

		/* Solve the linear system for the i^th column of the inverse matrix: :  */
		LU_substitution_6D(Amtmp, dxm, permute);

		for (j = 0; j < n; j++) { Aminv[j][i] = dxm[j]; }

	}

	return(0);
}

//4D LU-decomposition
__device__ int LU_decompose(double A[][NDIM], int permute[]){
	double row_norm[NDIM];
	double  absmax, maxtemp;
	int i, j, k, max_row;

	max_row = 0;

	//Find the maximum elements per row so that we can pretend late we have unit-normalized each equation

	for (i = 0; i < NDIM; i++) {
		absmax = 0.;

		for (j = 0; j < NDIM; j++) {
			maxtemp = fabs(A[i][j]);
			if(!isfinite((A[i][j]))) return(1);
			absmax = MY_MAX(absmax, maxtemp);
		}

		//Make sure that there is at least one non-zero element in this row:
		if (absmax == 0.) return(1);

		row_norm[i] = 1. / absmax; //Set the row's normalization factor.
	}

	/* For each of the columns, starting from the left ... */
	for (j = 0; j < NDIM; j++) {

		/* For each of the rows starting from the top.... */

		/* Calculate the Upper part of the matrix:  i < j :   */
		for (i = 0; i < j; i++) {
			for (k = 0; k < i; k++) A[i][j] -= A[i][k] * A[k][j];
		}

		absmax = 0.0;

		/* Calculate the Lower part of the matrix:  i <= j :   */
		for (i = j; i < NDIM; i++) {
			for (k = 0; k < j; k++) A[i][j] -= A[i][k] * A[k][j];

			maxtemp = fabs(A[i][j]) * row_norm[i];

			if (maxtemp >= absmax) {
				absmax = maxtemp;
				max_row = i;
			}

		}

		if (max_row != j) {
		
			if ((j == (NDIM - 2)) && (A[j][j + 1] == 0.)) max_row = j;
			else {
				for (k = 0; k < NDIM; k++) {
					maxtemp = A[j][k];
					A[j][k] = A[max_row][k];
					A[max_row][k] = maxtemp;
				}

				row_norm[max_row] = row_norm[j];
			}
		}

		permute[j] = max_row;

		if (A[j][j] == 0.) A[j][j] = 1.e-30;

		if (j != (NDIM - 1)) {
			maxtemp = 1. / A[j][j];

			for (i = (j + 1); i < NDIM; i++) A[i][j] *= maxtemp;
		}
	}

	return(0);
}

//5D LU-decomposition
__device__ int LU_decompose_3D(double A[][3], int permute[])
{
	double row_norm[3];
	double absmin = 1.e-30; /* Value used instead of 0 for singular matrices */
	double  absmax, maxtemp;
	int i, j, k, max_row;
	int n = 3;

	max_row = 0;
	for (i = 0; i < n; i++) {
		absmax = 0.;

		for (j = 0; j < n; j++) {

			maxtemp = fabs(A[i][j]);
			if (!isfinite((A[i][j]))) return(1);
			absmax = MY_MAX(absmax, maxtemp);
		}

		if (absmax == 0.) {
			return(1);
		}

		row_norm[i] = 1. / absmax;   /* Set the row's normalization factor. */
	}

	for (j = 0; j < n; j++) {
		for (i = 0; i < j; i++) {
			for (k = 0; k < i; k++) {
				A[i][j] -= A[i][k] * A[k][j];
			}
		}

		absmax = 0.0;

		for (i = j; i < n; i++) {
			for (k = 0; k < j; k++) {
				A[i][j] -= A[i][k] * A[k][j];
			}

			maxtemp = fabs(A[i][j]) * row_norm[i];

			if (maxtemp >= absmax) {
				absmax = maxtemp;
				max_row = i;
			}
		}

		if (max_row != j) {
			if ((j == (n - 2)) && (A[j][j + 1] == 0.)) {
				max_row = j;
			}
			else {
				for (k = 0; k < n; k++) {

					maxtemp = A[j][k];
					A[j][k] = A[max_row][k];
					A[max_row][k] = maxtemp;

				}
				row_norm[max_row] = row_norm[j];
			}
		}

		permute[j] = max_row;

		if (A[j][j] == 0.) {
			A[j][j] = absmin;
		}

		if (j != (n - 1)) {
			maxtemp = 1. / A[j][j];

			for (i = (j + 1); i < n; i++) {
				A[i][j] *= maxtemp;
			}
		}

	}

	return(0);

	/* End of LU_decompose() */
}

//5D LU-decomposition
__device__ int LU_decompose_5D(double A[][5], int permute[])
{
	double row_norm[5];
	double absmin = 1.e-30; /* Value used instead of 0 for singular matrices */
	double  absmax, maxtemp;
	int i, j, k, max_row;
	int n = 5;

	max_row = 0;
	for (i = 0; i < n; i++) {
		absmax = 0.;

		for (j = 0; j < n; j++) {

			maxtemp = fabs(A[i][j]);
			if (!isfinite((A[i][j]))) return(1);
			absmax = MY_MAX(absmax, maxtemp);
		}

		if (absmax == 0.) {
			return(1);
		}

		row_norm[i] = 1. / absmax;   /* Set the row's normalization factor. */
	}

	for (j = 0; j < n; j++) {
		for (i = 0; i < j; i++) {
			for (k = 0; k < i; k++) {
				A[i][j] -= A[i][k] * A[k][j];
			}
		}

		absmax = 0.0;

		for (i = j; i < n; i++) {
			for (k = 0; k < j; k++) {
				A[i][j] -= A[i][k] * A[k][j];
			}

			maxtemp = fabs(A[i][j]) * row_norm[i];

			if (maxtemp >= absmax) {
				absmax = maxtemp;
				max_row = i;
			}
		}

		if (max_row != j) {
			if ((j == (n - 2)) && (A[j][j + 1] == 0.)) {
				max_row = j;
			}
			else {
				for (k = 0; k < n; k++) {

					maxtemp = A[j][k];
					A[j][k] = A[max_row][k];
					A[max_row][k] = maxtemp;

				}
				row_norm[max_row] = row_norm[j];
			}
		}

		permute[j] = max_row;

		if (A[j][j] == 0.) {
			A[j][j] = absmin;
		}

		if (j != (n - 1)) {
			maxtemp = 1. / A[j][j];

			for (i = (j + 1); i < n; i++) {
				A[i][j] *= maxtemp;
			}
		}

	}

	return(0);

	/* End of LU_decompose() */
}

//6D LU-decomposition
__device__ int LU_decompose_6D(double A[][6], int permute[])
{
	double row_norm[6];
	double absmin = 1.e-30; /* Value used instead of 0 for singular matrices */
	double  absmax, maxtemp;
	int i, j, k, max_row;
	int n = 6;

	max_row = 0;
	for (i = 0; i < n; i++) {
		absmax = 0.;

		for (j = 0; j < n; j++) {

			maxtemp = fabs(A[i][j]);
			if (!isfinite((A[i][j]))) return(1);
			absmax = MY_MAX(absmax, maxtemp);
		}

		if (absmax == 0.) {
			return(1);
		}

		row_norm[i] = 1. / absmax;   /* Set the row's normalization factor. */
	}

	for (j = 0; j < n; j++) {
		for (i = 0; i < j; i++) {
			for (k = 0; k < i; k++) {
				A[i][j] -= A[i][k] * A[k][j];
			}
		}

		absmax = 0.0;

		for (i = j; i < n; i++) {
			for (k = 0; k < j; k++) {
				A[i][j] -= A[i][k] * A[k][j];
			}

			maxtemp = fabs(A[i][j]) * row_norm[i];

			if (maxtemp >= absmax) {
				absmax = maxtemp;
				max_row = i;
			}
		}

		if (max_row != j) {
			if ((j == (n - 2)) && (A[j][j + 1] == 0.)) {
				max_row = j;
			}
			else {
				for (k = 0; k < n; k++) {

					maxtemp = A[j][k];
					A[j][k] = A[max_row][k];
					A[max_row][k] = maxtemp;

				}
				row_norm[max_row] = row_norm[j];
			}
		}

		permute[j] = max_row;

		if (A[j][j] == 0.) {
			A[j][j] = absmin;
		}

		if (j != (n - 1)) {
			maxtemp = 1. / A[j][j];

			for (i = (j + 1); i < n; i++) {
				A[i][j] *= maxtemp;
			}
		}

	}

	return(0);

	/* End of LU_decompose() */
}

__device__ void LU_substitution(double A[][NDIM], double B[], int permute[])
{
	int i, j;
	double tmpvar;

	/* Perform the forward substitution using the LU matrix.
	*/
	for (i = 0; i < NDIM; i++) {
		tmpvar = B[permute[i]];
		B[permute[i]] = B[i];
		for (j = (i - 1); j >= 0; j--) {
			tmpvar -= A[i][j] * B[j];
		}
		B[i] = tmpvar;
	}

	/* Perform the backward substitution using the LU matrix.
	*/
	for (i = (NDIM - 1); i >= 0; i--) {
		for (j = (i + 1); j < NDIM; j++) {
			B[i] -= A[i][j] * B[j];
		}
		B[i] /= A[i][i];
	}
}


//3D LU-Substitution
__device__ void LU_substitution_3D(double A[][3], double B[], int permute[])
{
	int i, j;
	int n = 3;
	double tmpvar;

	for (i = 0; i < n; i++) {
		tmpvar = B[permute[i]];
		B[permute[i]] = B[i];
		for (j = (i - 1); j >= 0; j--) {
			tmpvar -= A[i][j] * B[j];
		}
		B[i] = tmpvar;
	}

	for (i = (n - 1); i >= 0; i--) {
		for (j = (i + 1); j < n; j++) {
			B[i] -= A[i][j] * B[j];
		}
		B[i] /= A[i][i];
	}
}

//5D LU-Substitution
__device__ void LU_substitution_5D(double A[][5], double B[], int permute[])
{
	int i, j;
	int n = 5;
	double tmpvar;

	for (i = 0; i < n; i++) {
		tmpvar = B[permute[i]];
		B[permute[i]] = B[i];
		for (j = (i - 1); j >= 0; j--) {
			tmpvar -= A[i][j] * B[j];
		}
		B[i] = tmpvar;
	}

	for (i = (n - 1); i >= 0; i--) {
		for (j = (i + 1); j < n; j++) {
			B[i] -= A[i][j] * B[j];
		}
		B[i] /= A[i][i];
	}
}

//6D LU-Substitution
__device__ void LU_substitution_6D(double A[][6], double B[], int permute[])
{
	int i, j;
	int n = 6;
	double tmpvar;

	for (i = 0; i < n; i++) {
		tmpvar = B[permute[i]];
		B[permute[i]] = B[i];
		for (j = (i - 1); j >= 0; j--) {
			tmpvar -= A[i][j] * B[j];
		}
		B[i] = tmpvar;
	}

	for (i = (n - 1); i >= 0; i--) {
		for (j = (i + 1); j < n; j++) {
			B[i] -= A[i][j] * B[j];
		}
		B[i] /= A[i][i];
	}
}