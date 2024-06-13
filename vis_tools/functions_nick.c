#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
//do not include malloc.h and omp.h under Mac OS X since they are not supported
#if(!defined(__APPLE__))
#include <malloc.h>
#include <omp.h>
#endif



void calc_twist(int bs1, int bs2, int bs3, int nb, float *r, float *h, float *ph, float *rho, float *tilt, float *prec);
void kernel_rdump_new_1block(int n, int flag, int RAD_M1, char *dir, int dump, int n_active_total, int f1,int f2,int f3, int nb, int bs1, int bs2, int bs3, float* rho,float* ug, float* uu, float* B,float* E_rad,float* uu_rad,float* gcov,float* gcon, int axisym);
void kernel_rgdump_new_1block(int flag, char *dir, int axisym, int *n_ord, int f1, int f2, int f3, int nb, int bs1, int bs2, int bs3, float* x1,float* x2,float* x3,float* r,float* h,float* ph,float* gcov,float* gcon,float* dxdxp,float* gdet);


/*
void kernel_rgdump_new_1block()

This is the equivalent of kernel_rgdump_new() designed to only operate on 1 block. It is functionally the same except the first argument is n, which is the index of the block that is read. nb is the total number of blocks. 
*/
void kernel_rgdump_new_1block(int n, int flag, char *dir, int axisym, int *n_ord, int f1, int f2, int f3, int nb, int bs1, int bs2, int bs3, float* x1,float* x2,float* x3,float* r,float* h,float* ph,float* gcov,float* gcon,float* dxdxp,float* gdet){
    int n,i,j,z,k,filesize, ii, gridsize_2D, index_2D, gridsize_3D, index_3D;
    int float_size = sizeof(double);
    int bs1new=bs1/f1;
    int bs2new=bs2/f2;
    int bs3new=bs3/f3;
    char filename1[100],filename2[100];
    double trash[58];
    FILE *fin;

	if(flag){
        sprintf(filename2, "/gdumps/gdump%d", n_ord[n]);
        strcpy(filename1, dir);
        strcat(filename1,filename2);
        if( access(filename1, 0 ) != -1 ) {
            fin = fopen(filename1, "rb");
            fseek(fin, 0L, SEEK_END);
            filesize = ftell(fin);
            if(filesize==(58*bs1*bs2*bs3)*float_size){
                fseek(fin, 0L, 0);
                for(i=0;i<bs1new;i++)for(j=0;j<bs2new;j++)for(z=0;z<bs3new;z++){
                    index_3D=n*bs1new*bs2new*bs3new+i*bs3new*bs2new+j*bs3new+z;
                    gridsize_3D=nb*bs1new*bs2new*bs3new;
                    index_2D=n*bs1new*bs2new+i*bs2new+j;
                    gridsize_2D=nb*bs1new*bs2new;

                    fseek(fin, ((i*f1)*bs2*bs3*58+(j*f2)*bs3*58+(z*f3)*58)*float_size, SEEK_SET);
                    fread(&trash[0], float_size, 58, fin);

                    x1[index_3D]=trash[3];
                    x2[index_3D]=trash[4];
                    x3[index_3D]=trash[5];
                    r[index_3D]=trash[6];
                    h[index_3D]=trash[7];
                    ph[index_3D]=trash[8];
                    if(axisym){
                        if(z==0){
                            for(ii=0;ii<16;ii++)gcov[(ii)*gridsize_2D+index_2D]=trash[6+ii+3];
                            for(ii=0;ii<16;ii++)gcon[(ii)*gridsize_2D+index_2D]=trash[6+16+ii+3];
                            gdet[index_2D]=trash[6+32+3];
                            for(ii=0;ii<16;ii++)dxdxp[ii*gridsize_2D+index_2D]=trash[6+32+1+ii+3];
                        }
                    }
                    else{
                        for(ii=0;ii<16;ii++)gcov[(ii)*gridsize_3D+index_3D]=trash[6+ii+3];
                        for(ii=0;ii<16;ii++)gcon[(ii)*gridsize_3D+index_3D]=trash[6+16+ii+3];
                        gdet[index_3D]=trash[6+32+3];
                        for(ii=0;ii<16;ii++)dxdxp[ii*gridsize_3D+index_3D]=trash[6+32+1+ii+3];
                    }
                }
            }
            else{
                fprintf(stderr,"Possible data corruption (wrong size) in file: %s \n", filename1);
            }
            fclose(fin);
        }
        else{
            fprintf(stderr,"Possible data corruption in file: %s \n", filename1);
        }
    }
	else{
        sprintf(filename2, "/gdumps/gdump%d", n_ord[n]);
        strcpy(filename1, dir);
        strcat(filename1,filename2);
        if( access(filename1, 0 ) != -1 ) {
            fin = fopen(filename1, "rb");
            fseek(fin, 0L, SEEK_END);
            filesize = ftell(fin);
            if(filesize==(9*bs1*bs2*bs3+(bs1*bs2*49)*(axisym)+(bs1*bs2*bs3*49)*(!axisym))*float_size){
                fseek(fin, 0L, 0);
                for(i=0;i<bs1new;i++)for(j=0;j<bs2new;j++)for(z=0;z<bs3new;z++){
                    index_3D=n*bs1new*bs2new*bs3new+i*bs3new*bs2new+j*bs3new+z;
                    gridsize_3D=nb*bs1new*bs2new*bs3new;

                    fseek(fin, ((i*f1)*bs2*bs3*9+(j*f2)*bs3*9+(z*f3)*9)*float_size, SEEK_SET);
                    fread(&trash[0], float_size, 9, fin);

                    x1[index_3D]=trash[3];
                    x2[index_3D]=trash[4];
                    x3[index_3D]=trash[5];
                    r[index_3D]=trash[6];
                    h[index_3D]=trash[7];
                    ph[index_3D]=trash[8];
                }
                if(axisym){
                    for(i=0;i<bs1new;i++)for(j=0;j<bs2new;j++){
                        index_2D=n*bs1new*bs2new+i*bs2new+j;
                        gridsize_2D=nb*bs1new*bs2new;

                        fseek(fin, (9*bs1*bs2*bs3+(i)*bs2*49+(j)*49)*float_size, SEEK_SET);
                        fread(&trash[0], float_size,49, fin);

                        for(ii=0;ii<16;ii++) gcov[ii*gridsize_2D+index_2D]=trash[ii];
                        for(ii=0;ii<16;ii++) gcon[ii*gridsize_2D+index_2D]=trash[16+ii];
                        gdet[index_2D]=trash[32];
                        for(ii=0;ii<16;ii++) dxdxp[ii*gridsize_2D+index_2D]=trash[32+1+ii];
                    }
                }
                else{
                    for(i=0;i<bs1new;i++)for(j=0;j<bs2new;j++)for(z=0;z<bs3new;z++){
                        index_3D=n*bs1new*bs2new*bs3new+i*bs3new*bs2new+j*bs3new+z;
                        gridsize_3D=nb*bs1new*bs2new*bs3new;

                        fseek(fin, (9*bs1*bs2*bs3+(i*f1)*bs2*bs3*49+(j*f2)*bs3*49+(z*f3)*49)*float_size, SEEK_SET);
                        fread(&trash[0], float_size,49, fin);

                        for(ii=0;ii<16;ii++) gcov[ii*gridsize_3D+index_3D]=trash[ii];
                        for(ii=0;ii<16;ii++) gcon[ii*gridsize_3D+index_3D]=trash[16+ii];
                        gdet[index_3D]=trash[32];
                        for(ii=0;ii<16;ii++)dxdxp[ii*gridsize_3D+index_3D]=trash[32+1+ii];
                    }
                }
            }
            else{
                fprintf(stderr,"Possible data corruption (wrong size) in file: %s \n", filename1);
            }
            fclose(fin);
        }
        else{
            fprintf(stderr,"Possible data corruption in file: %s \n", filename1);
        }
    }
}

/*
void kernel_rdump_new_1block()

This is the equivalent of kernel_rdump_new() designed to only operate on 1 block. It is functionally the same except the first argument is n, which is the index of the block that is read. nb is the total number of blocks. 
*/
void kernel_rdump_new_1block(int n, int flag, int RAD_M1, char *directory, int dump, int n_active_total, int f1,int f2,int f3, int nb, int bs1, int bs2, int bs3, float* rho,float* ug, float* uu, float* B,float* E_rad,float* uu_rad,float* gcov,float* gcon, int axisym){
        int i,j,z,k, ii,u, gridsize_3D, index_3D, n_start,n_end, filesize;
        int float_size = sizeof(float);
        int bs1new=bs1/f1;
        int bs2new=bs2/f2;
        int bs3new=bs3/f3;
        int keep_looping, num_threads, thread_id;
        int NPR=9+5*RAD_M1;
        char filename1[100],filename2[100];
        float trash[14];
        FILE *fin;

        keep_looping=1;
        u=0;
        #if(defined(_OPENMP))
            num_threads = omp_get_num_threads();
	    thread_id = omp_get_thread_num();
	#else
	    num_threads = 1;
	    thread_id = 0;
	#endif
	

        while(keep_looping){
            if(flag)sprintf(filename2, "/dumps%d/new_dump", dump);
            else sprintf(filename2, "/dumps%d/new_dump%d", dump, u);
            strcpy(filename1, directory);
            strcat(filename1,filename2);

            if( access(filename1, 0 ) != -1 ) {
                fin = fopen(filename1, "rb");
                fseek(fin, 0L, SEEK_END);
                filesize = ftell(fin);
                if(filesize%(NPR*bs1*bs2*bs3*float_size)==0){
                    fseek(fin, 0L, 0);
                    for(i=0;i<bs1new;i++)for(j=0;j<bs2new;j++)for(z=0;z<bs3new;z++){
                        index_3D=n*bs1new*bs2new*bs3new+i*bs3new*bs2new+j*bs3new+z;
                        gridsize_3D=nb*bs1new*bs2new*bs3new;

                        fseek(fin, (n*NPR*bs1*bs2*bs3+(i*f1)*bs2*bs3*NPR+(j*f2)*bs3*NPR+(z*f3)*NPR)*float_size, SEEK_SET);
                        fread(&trash[0], float_size, NPR, fin);

                        rho[index_3D]=trash[0];
                        ug[index_3D]=trash[1];
                        for(ii=0;ii<4;ii++)uu[(ii)*gridsize_3D+index_3D]=trash[2+ii];
                        for(ii=0;ii<3;ii++)B[(ii+1)*gridsize_3D+index_3D]=trash[6+ii];

                        //Radiation variables
                        if(RAD_M1==1){
                            E_rad[index_3D]=trash[9];
                            for(ii=1;ii<4;ii++) uu_rad[(ii)*gridsize_3D+index_3D]=trash[10+ii];
                        }
                    }
                    if(n==n_active_total-1){
                        keep_looping=0;
                        break;
                    }
                }
                else{
                    fprintf(stderr,"Possible data corruption (wrong size) in file: %s \n", filename1);
                }
                fclose(fin);

                u++;
                if(flag) keep_looping=0;
            }
            else keep_looping=0;
        }
}







void calc_twist(int bs1, int bs2, int bs3, int nb, float *r, float *h, float *ph, float *rho, float *tilt, float *prec){
	int rr, hh, pp, n;
	float rho_avg;
	int   tilt_index, index_3D;



	for (n=0; n<nb; n++){
		//#pragma omp parallel for private(threadid,dxdr,Tuu,Tuu_tmp,xc,uu_local,bu_local,B_local,gcov_local, gcon_local, i,j, z, i1, j1, mu, nu,k, l, index_2D, index_3D, gridsize_3D, gridsize_2D)
		for (rr = 0; rr<bs1; rr++){
			tilt[rr] = 0;
			rho_avg  = 0;
			for (hh = 0; hh < bs2; hh++) for (pp = 0; pp < bs3; pp++){
				index_3D=n*bs1*bs2*bs3 + rr*bs3*bs2 + hh*bs3 + pp; // index for flattened arrays

				tilt[rr] += rho[index_3D]*fabs(cos(h[index_3D])); // weight in rho to find midplane of disk; take fabs because disk theta changes sign across phi when tilted; distribution uniform in cos(theta)
				rho_avg  += rho[index_3D]; // sum density to normalize tilt
				tilt[rr] = tilt[rr] / rho_avg;
				tilt[rr] = acos(tilt[rr]); // get tilt angle
			} // bs2/bs3 loop

			
			// find the index corresponding to the tilt of the disk
			// we can use this to find the precession angle
			tilt_index = bs2-1;
			for (hh = 0; hh < bs2; hh++){
				if ((h[hh]-M_PI/2) > tilt[rr]){
					tilt_index = hh-1;
					break;
				}
			} // for bs2 loop 
			/* Solve for precession angle by taking density-weighted average phi.
			   This is done at theta ~ tilt, because in the mid-plane there are two locations where the density peaks (i.e., the nodes)
			   At the tilt angle, there is one location where density peaks. 
			   There may be a better way to do this */
			prec[rr]    = 0; 
			rho_avg     = 0;
			for (pp = 0; pp < bs3; pp++){
			    index_3D=n*bs1*bs2*bs3 + rr*bs3*bs2 + tilt_index*bs3 + pp; // index for flattened array at tilt_index

			    prec[rr] += rho[index_3D]*ph[index_3D];
			    rho_avg  += rho[index_3D]; // sum density to normalize prec
			} // bs3 loop
			prec[rr] = prec[rr] / rho_avg;
		} // bs1 loop
	} // nb loop
}
