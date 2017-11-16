#include "decsCUDA.h"
extern "C" {
#include "decs.h"
}


void GPU_hcor(int n){

}

void GPU_fluxcalcprep(int dir, int flag, int ppm_solver, int n)
{
	/*Set arguments of kernel*/
	if (dir == 1){
		if (flag == 1){
			 fluxcalcprep << < nr_workgroups2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF1_1[n], Bufferdq_1[n], Bufferph_1[n], dir, lim, ppm_solver, BufferV[n]);
		}
		else{
			 fluxcalcprep << < nr_workgroups2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF1_1[n], Bufferdq_1[n], Bufferp_1[n], dir, lim, ppm_solver, BufferV[n]);
		}
	}
	else if (dir == 2){
		if (flag == 1){
			 fluxcalcprep << < nr_workgroups2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF2_1[n], Bufferdq_1[n], Bufferph_1[n], dir, lim, ppm_solver, BufferV[n]);
		}
		else{
			 fluxcalcprep << < nr_workgroups2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF2_1[n], Bufferdq_1[n], Bufferp_1[n], dir, lim, ppm_solver, BufferV[n]);
		}
	}
	else{
		if (flag == 1){
			 fluxcalcprep << < nr_workgroups2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF3_1[n], Bufferdq_1[n], Bufferph_1[n], dir, lim, ppm_solver, BufferV[n]);
		}
		else{
			 fluxcalcprep << < nr_workgroups2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF3_1[n], Bufferdq_1[n], Bufferp_1[n], dir, lim, ppm_solver, BufferV[n]);
		}
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status ) printf("Error Fluxcalcprep %d \n", status);
}

void GPU_fluxcalc2D(int dir, int flag, int n)
{
	////cudaSetDevice(block[n][AMR_GPU]);
	/*Calculate reconstructed left state*/
	GPU_fluxcalcprep(dir, flag, 1, n);
	if (flag == 1){

		if (dir == 1){
			fluxcalc2D2 << < nr_workgroups2_1[n], local_work_size[0], local_work_size[0] * sizeof(double), commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF1_1[n], Bufferdq_1[n], Bufferph_1[n], Bufferpsh_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n],
				lim, dir, gam, cour, dtij_GPU[n], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3), Bufferstorage1[n], 
				Bufferstorage2[n], Bufferstorage3[n], Bufferstorage4[n], dx[n][1], dx[n][2], dx[n][3]);
		}
		if (dir == 2){
			fluxcalc2D2 << < nr_workgroups2_2[n], local_work_size[0], local_work_size[0] * sizeof(double), commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF2_1[n], Bufferdq_1[n], Bufferph_1[n], Bufferpsh_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n],
				 lim, dir, gam, cour, dtij_GPU[n], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3), Bufferstorage1[n],
				Bufferstorage2[n], Bufferstorage3[n], Bufferstorage4[n], dx[n][1], dx[n][2], dx[n][3]);
		}
		if (dir == 3){
			fluxcalc2D2 << < nr_workgroups2_3[n], local_work_size[0], local_work_size[0] * sizeof(double), commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF3_1[n], Bufferdq_1[n], Bufferph_1[n], Bufferpsh_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n],
				 lim, dir, gam, cour, dtij_GPU[n], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3), Bufferstorage1[n],
				Bufferstorage2[n], Bufferstorage3[n], Bufferstorage4[n], dx[n][1], dx[n][2], dx[n][3]);
		}
	}
	else{
		if (dir == 1){
			fluxcalc2D2 << < nr_workgroups2_1[n], local_work_size[0], local_work_size[0] * sizeof(double), commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF1_1[n], Bufferdq_1[n], Bufferp_1[n], Bufferps_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n],
				 lim, dir, gam, cour, dtij_GPU[n], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3), Bufferstorage1[n],
				Bufferstorage2[n], Bufferstorage3[n], Bufferstorage4[n], dx[n][1], dx[n][2], dx[n][3]);
		}
		if (dir == 2){
			fluxcalc2D2 << < nr_workgroups2_2[n], local_work_size[0], local_work_size[0] * sizeof(double), commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF2_1[n], Bufferdq_1[n], Bufferp_1[n], Bufferps_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n],
				 lim, dir, gam, cour, dtij_GPU[n], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3), Bufferstorage1[n],
				Bufferstorage2[n], Bufferstorage3[n], Bufferstorage4[n], dx[n][1], dx[n][2], dx[n][3]);
		}
		if (dir == 3){
			fluxcalc2D2 << < nr_workgroups2_3[n], local_work_size[0], local_work_size[0] * sizeof(double), commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF3_1[n], Bufferdq_1[n], Bufferp_1[n], Bufferps_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n],
				 lim, dir, gam, cour, dtij_GPU[n], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3), Bufferstorage1[n],
				Bufferstorage2[n], Bufferstorage3[n], Bufferstorage4[n], dx[n][1], dx[n][2], dx[n][3]);
		}
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status) printf("Error Fluxcalc2D2 %d\n", status);

	/*Calculate reconstructed right state*/
	#if(PPM || LEER)
	GPU_fluxcalcprep(dir, flag, 2, n);
	printf("PPM and Leer not yet fully implemented this way... \n");
	#endif
}

void GPU_fix_flux(int n)
{
	/*Run kernel*/
	//cudaSetDevice(block[n][AMR_GPU]);
	 fix_flux << < nr_workgroups_special[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], block[n][AMR_NBR1], block[n][AMR_NBR2], block[n][AMR_NBR3], block[n][AMR_NBR4]);
	// cudaDeviceSynchronize();
	 status = cudaGetLastError();
	if (cudaSuccess != status)printf("Error fixflux %d \n", status);
}

void GPU_consttransport1(int flag, double Dt, int n){
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((N1_GPU[n] + N1G) * (N2_GPU[n] + N2G) * (N3_GPU[n] + N3G)) % LOCAL_WORK_SIZE) + (N1_GPU[n] + N1G) * (N2_GPU[n] + N2G) * (N3_GPU[n] + N3G)) / LOCAL_WORK_SIZE;

	//cudaSetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	if (flag == 1){
		 consttransport1 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], Bufferph_1[n], Bufferstorage1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n]);
	}
	else{
		 consttransport1 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], Bufferp_1[n], Bufferstorage1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n]);
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status)printf("Error consttransport1 %d \n", status);
}

void GPU_consttransport2(int flag, double Dt, int n){
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((N1_GPU[n] + D1) * (N2_GPU[n] + D2) * (N3_GPU[n] + D3)) % LOCAL_WORK_SIZE) + (N1_GPU[n] + D1) * (N2_GPU[n] + D2) * (N3_GPU[n] + D3)) / LOCAL_WORK_SIZE;

	//cudaSetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	if (flag == 1){
		 consttransport2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferE_1[n], Bufferstorage1[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], 
			Bufferph_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3));
	}
	else{
		 consttransport2 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferE_1[n], Bufferstorage1[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], 
			Bufferp_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], block[n][AMR_NBR1]<0 || (block[n][AMR_POLE] == 1 || block[n][AMR_POLE] == 3), block[n][AMR_NBR3]<0 || (block[n][AMR_POLE] == 2 || block[n][AMR_POLE] == 3));
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status)printf("Error constransport2 %d \n", status);
}

void GPU_consttransport3(int flag, double Dt, int n){
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((N1_GPU[n] + D1) * (N2_GPU[n] + D2) * (N3_GPU[n] + D3)) % LOCAL_WORK_SIZE) + (N1_GPU[n] + D1) * (N2_GPU[n] + D2) * (N3_GPU[n] + D3)) / LOCAL_WORK_SIZE;

	//cudaSetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	if (flag == 1){
		 consttransport3 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], dx[n][1], dx[n][2], dx[n][3], Buffergdet[n], Bufferps_1[n], Bufferps_1[n], BufferE_1[n], Dt);
	}
	else{
		 consttransport3 << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], dx[n][1], dx[n][2], dx[n][3], Buffergdet[n], Bufferps_1[n], Bufferpsh_1[n], BufferE_1[n], Dt);
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status )printf("Error constransport3 %d \n", status);
}

void GPU_flux_ct1(int n)
{
	//cudaSetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	 flux_ct1 << < nr_workgroups3[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], Bufferdq_1[n]);
	// cudaDeviceSynchronize();
	 status = cudaGetLastError();
	if (cudaSuccess != status ) printf("Error fluxct1 %d\n", status);
}

void GPU_flux_ct2(int n)
{
	//cudaSetDevice(block[n][AMR_GPU]);
	/*Run kernel*/
	 flux_ct2 << < nr_workgroups3[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], Bufferdq_1[n]);
	 //cudaDeviceSynchronize();
	 status = cudaGetLastError();
	if (cudaSuccess != status ) printf("Error fluxct2 %d\n", status);
}

void GPU_Utoprim(int flag, int n, double Dt)
{
	//cudaSetDevice(block[n][AMR_GPU]);
	if (flag == 0){
		//Utoprim0 << < nr_workgroups1[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], Bufferp_1[n], Bufferp_1[n], Bufferph_1[n], Bufferpsh_1[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], Bufferdq_1[n],
		//	Bufferradius[n], Bufferpflag[n], Bufferfailimage[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], Bufferconn[n], BufferKatm[n], gam, dx[n][1], dx[n][2], dx[n][3], a, Dt, flag);
	}
	else{
		Utoprim0 << < nr_workgroups1[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], Bufferp_1[n], Bufferph_1[n], Bufferp_1[n], Bufferps_1[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], Bufferdq_1[n],
			Bufferradius[n], Bufferpflag[n], Bufferfailimage[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], Bufferconn[n], BufferKatm[n], gam, dx[n][1], dx[n][2], dx[n][3], a, Dt, flag);
	}
	status = cudaGetLastError();
	if (cudaSuccess != status) printf("Error Utoprim0 %d\n", status);

	if (flag == 0){
		//Utoprim1 << < nr_workgroups1[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], Bufferp_1[n], Bufferp_1[n], Bufferph_1[n], Bufferpsh_1[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n],
		//	Bufferradius[n], Bufferpflag[n], Bufferfailimage[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], Bufferconn[n], BufferKatm[n], gam, dx[n][1], dx[n][2], dx[n][3], a, Dt, flag);
	}
	status = cudaGetLastError();
	if (cudaSuccess != status) printf("Error Utoprim1 %d\n", status);
	if (flag == 0){
		//Utoprim2 << < nr_workgroups1[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], Bufferp_1[n], Bufferph_1[n], Bufferph_1[n], Bufferpsh_1[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], Bufferdq_1[n],
		//	Bufferradius[n], Bufferpflag[n], Bufferfailimage[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], Bufferconn[n], BufferKatm[n], gam, dx[n][1], dx[n][2], dx[n][3], a, Dt, flag);
	}
	else{
		Utoprim2 << < nr_workgroups1[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], Bufferp_1[n], Bufferph_1[n], Bufferp_1[n], Bufferps_1[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], Bufferdq_1[n],
			Bufferradius[n], Bufferpflag[n], Bufferfailimage[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], Bufferconn[n], BufferKatm[n], gam, dx[n][1], dx[n][2], dx[n][3], a, Dt, flag);
	}
	status = cudaGetLastError();
	if (cudaSuccess != status) printf("Error Utoprim1 %d\n", status);
}

void GPU_fixuputoprim(int flag, int n)
{
	int nr_workgroups_local[1];
	nr_workgroups_local[0] = ((LOCAL_WORK_SIZE - ((N1_GPU[n] + 2 * N1G) * (N2_GPU[n] + 2 * N2G) * (N3_GPU[n] + 2 * N3G)) % LOCAL_WORK_SIZE) + (N1_GPU[n] + 2 * N1G) * (N2_GPU[n] + 2 * N2G) * (N3_GPU[n] + 2 * N3G)) / LOCAL_WORK_SIZE;
	//cudaSetDevice(block[n][AMR_GPU]);
	if (flag == 1){
		 fixuputoprim << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], Bufferp_1[n], Bufferpflag[n], Bufferfailimage[n], Buffergcov[n], Buffergcon[n], Buffergdet[n]);
	}
	else{
		 fixuputoprim << < nr_workgroups_local[0], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], Bufferph_1[n], Bufferpflag[n], Bufferfailimage[n], Buffergcov[n], Buffergcon[n], Buffergdet[n]);
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status ) printf("Error fixuputoprim %d\n", status);
}

void GPU_fixup(int flag, int n, double Dt)
{
	//cudaSetDevice(block[n][AMR_GPU]);
	if (flag == 0){
		fixup << < nr_workgroups1[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], Bufferp_1[n], Bufferp_1[n], Bufferph_1[n], Bufferpsh_1[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], Bufferdq_1[n],
			Bufferradius[n], Bufferpflag[n], Bufferfailimage[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], Bufferconn[n], BufferKatm[n], gam, dx[n][1], dx[n][2], dx[n][3], a, Dt, flag);
	}
	else{
		fixup << < nr_workgroups1[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], Bufferp_1[n], Bufferph_1[n], Bufferp_1[n], Bufferps_1[n], BufferF1_1[n], BufferF2_1[n], BufferF3_1[n], Bufferdq_1[n],
			Bufferradius[n], Bufferpflag[n], Bufferfailimage[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], Bufferconn[n], BufferKatm[n], gam, dx[n][1], dx[n][2], dx[n][3], a, Dt, flag);
	}
	//cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (cudaSuccess != status ) printf("Error fixup %d\n", status);
}


void GPU_boundprim1(int flag, int n)
{
	if (block[n][AMR_NBR2] == -1 || block[n][AMR_NBR4] == -1){
		if (flag == 0){
			 boundprim1 << < nr_workgroups_special1[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], Bufferph_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], block[n][AMR_NBR2], block[n][AMR_NBR4], Bufferpsh_1[n]);
		}
		else{
			 boundprim1 << < nr_workgroups_special1[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], Bufferp_1[n], Buffergcov[n], Buffergcon[n], Buffergdet[n], block[n][AMR_NBR2], block[n][AMR_NBR4], Bufferps_1[n]);
		}
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (cudaSuccess != status ) printf("Error boundprim1 %d\n", status);
	}
}

void GPU_boundprim2(int flag, int n)
{
	if (block[n][AMR_NBR1] == -1 || block[n][AMR_NBR3] == -1){
		if (flag == 0){
			 boundprim2 << < nr_workgroups_special2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], Bufferph_1[n], Buffergdet[n], block[n][AMR_NBR1], block[n][AMR_NBR3], Bufferpsh_1[n]);
		}
		else{
			 boundprim2 << < nr_workgroups_special2[n], local_work_size[0], 0, commandQueueGPU[n] >> > (N1_GPU[n], N2_GPU[n], N3_GPU[n], Bufferp_1[n], Buffergdet[n], block[n][AMR_NBR1], block[n][AMR_NBR3], Bufferps_1[n]);
		}
		//cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (cudaSuccess != status) printf("Error boundprim2.1 %d\n", status);
	}
}
