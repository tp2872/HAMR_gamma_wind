#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
//do not include malloc.h and omp.h under Mac OS X since they are not supported
#if(!defined(__APPLE__))
#include <malloc.h>
#include <omp.h>
#endif
//#include <unistd.h>

void calc_cart_c(float r, float h, float ph, float* dxdr);
void lower_c(float *uu, float *ud, float *gcov);
void bu_calc(float *B, float *uu, float *bu, float *gcov);
void eu_calc(float *E, float *uu, float *eu, float *gcov);
void Tcalcuu_c(float rho, float ug, float *uu, float *bu, float *gcov, float *gcon, float gam, float *Tuu);
void Tcalcud_c(float rho, float ug, float *uu, float *bu, float *gcov, float *gcon, float gam, float *Tud);
void kernel_calc_prec_disk(int bs1, int bs2, int bs3, int nb, int axisym, int avg, float *r, float *h, float *ph, float *rho, float *ug, float *uu, float *B, float gam, float* gcov, float* gcon, float* gdet, float* dxdxp,
float *Su_disk, float *L_disk,float *Su_corona, float *L_corona,float *Su_disk_avg, float *L_disk_avg,float *Su_corona_avg, float *L_corona_avg);
void kernel_mdot(int bs1, int bs2, int bs3, int nb, int a_ndim, int b_ndim, float *a, float *b, float *c);
void kernel_misc_calc(int bs1, int bs2, int bs3, int nb, int axisym, float *uu, float *B, float *E, float *bu,float *eu, float *gcov, float *bsq, float *esq, int calc_bu, int calc_eu,int calc_bsq, int calc_esq);
void kernel_rdump_new(int flag, int RAD_M1,int RESISTIVE,int TWO_T, int P_NUM, char *dir, int dump, int n_active_total, int f1,int f2,int f3, int nb, int bs1, int bs2, int bs3, float* rho,float* ug, float* uu, float* B,float* E,float* E_rad,float* uu_rad,float* TE,float* TI,float* photon_number,float* gcov,float* gcon, int axisym);
void kernel_rgdump_new(int flag, char *dir,  int axisym, int *n_ord, int f1, int f2, int f3, int nb, int bs1, int bs2, int bs3, float* x1,float* x2,float* x3,float* r,float* h,float* ph,float* gcov,float* gcon,float* dxdxp,float* gdet);
void kernel_rdump_write(int flag, int RAD_M1,int RESISTIVE,int TWO_T, int P_NUM, char *dir, int dump, int n_active_total, int f1,int f2,int f3, int nb, int bs1, int bs2, int bs3, float* rho,float* ug, float* uu, float* B,float* E,float* E_rad,float* uu_rad,float* TE,float* TI,float* photon_number,float* gcov,float* gcon, int axisym);
void kernel_rgdump_write(int flag, char *dir,  int axisym, int *n_ord, int f1, int f2, int f3, int nb, int bs1, int bs2, int bs3, float* x1,float* x2,float* x3,float* r,float* h,float* ph,float* gcov,float* gcon,float* dxdxp,float* gdet);
void kernel_griddata3D_new(int nb, int bs1new, int bs2new, int bs3new, int nb1, int nb2, int nb3, int* n_ord, float* input, float* output, int ACTIVE1, int ACTIVE2, int ACTIVE3, int * AMR_LEVEL1, int * AMR_LEVEL2, int * AMR_LEVEL3, int * AMR_COORD1, int * AMR_COORD2, int * AMR_COORD3);
void kernel_griddata2D_new(int nb, int bs1new, int bs2new, int bs3new, int nb1, int nb2, int nb3, int* n_ord, float* input, float* output, int ACTIVE1, int ACTIVE2, int ACTIVE3, int * AMR_LEVEL1, int * AMR_LEVEL2, int * AMR_LEVEL3, int * AMR_COORD1, int * AMR_COORD2, int * AMR_COORD3);
void kernel_rgdump_griddata(int flag, int interpolate,char *dir, int axisym, int *n_ord, int f1, int f2, int f3, int nb, int bs1, int bs2, int bs3, float* x1,float* x2,float* x3,float* r,float* h,float* ph,float* gcov,float* gcon,float* dxdxp,float* gdet,int ACTIVE1,
int ACTIVE2, int ACTIVE3, int * AMR_LEVEL1, int * AMR_LEVEL2, int * AMR_LEVEL3, int * AMR_COORD1, int * AMR_COORD2, int * AMR_COORD3, int nb1, int nb2, int nb3, int REF_1, int REF_2, int REF_3,  float startx1, float startx2, float startx3, float _dx1, float _dx2, float _dx3, int export_raytracing, int i_min, int i_max, int j_min, int j_max, int z_min, int z_max);
void kernel_rdump_griddata(int flag, int interpolate,int RAD_M1,int RESISTIVE,int TWO_T, int P_NUM, char *dir, int dump, int n_active_total, int f1,int f2,int f3, int nb, int bs1, int bs2, int bs3, float* rho,float* ug, float* uu, float* B,float* E,float* E_rad,float* uu_rad,float* TE,float* TI,float* photon_number,float* gcov,float* gcon,
int axisym,int* n_ord,int ACTIVE1, int ACTIVE2, int ACTIVE3, int * AMR_LEVEL1, int * AMR_LEVEL2, int * AMR_LEVEL3, int * AMR_COORD1, int * AMR_COORD2, int * AMR_COORD3, int nb1, int nb2, int nb3,
int REF_1, int REF_2, int REF_3, int export_raytracing, float DISK_THICKNESS, float a, float gam, float* Rdot, float* bsq, float* r, float startx1, float startx2, float startx3, float _dx1, float _dx2, float _dx3, float* x1, float* x2, float* x3, int i_min, int i_max, int j_min, int j_max, int z_min, int z_max);
float slope_lim(float y1, float y2, float y3, float x1, float x2, float x3);
void invert_4x4(double *a, double *b);
void kernel_invert_4x4(float *A, float *B, int nb, int bs1, int bs2, int bs3);

float misc_source(float rho, float ug, float bsq, float ucon[4], float r, float gcov[4], float H_OVER_R, float a, float gam);

#define MIN(X,Y) ((X) < (Y) ?  (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ?  (X) : (Y))

void kernel_rgdump_new(int flag, char *dir, int axisym, int *n_ord, int f1, int f2, int f3, int nb, int bs1, int bs2, int bs3, float* x1,float* x2,float* x3,float* r,float* h,float* ph,float* gcov,float* gcon,float* dxdxp,float* gdet){
    int n,i,j,z,filesize, ii, gridsize_2D, index_2D, gridsize_3D, index_3D;
    int float_size = sizeof(double);
    int bs1new=bs1/f1;
    int bs2new=bs2/f2;
    int bs3new=bs3/f3;
    char filename1[100],filename2[100];
    double trash[58];
    FILE *fin;

	if(flag){
        #pragma omp parallel for private(n,i,j,z,filesize, ii,gridsize_2D, index_2D, gridsize_3D, index_3D, fin, filename1,filename2,trash)
        for(n=0;n<nb;n++){
            sprintf(filename2, "/gdumps/gdump%d", n_ord[n]);
            sprintf(filename1, dir);
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
    }
	else{
	   #pragma omp parallel for private(n,i,j,z,filesize, ii,gridsize_2D, index_2D, gridsize_3D, index_3D, fin, filename1,filename2, trash)
        for(n=0;n<nb;n++){
            sprintf(filename2, "/gdumps/gdump%d", n_ord[n]);
            sprintf(filename1, dir);
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
}

void kernel_rgdump_griddata(int flag,int interpolate, char *directory, int axisym, int *n_ord, int f1, int f2, int f3, int nb, int bs1, int bs2, int bs3, float* x1,float* x2,float* x3,float* r,float* h,float* ph,float* gcov,float* gcon,float* dxdxp,float* gdet,
int ACTIVE1, int ACTIVE2, int ACTIVE3, int * AMR_LEVEL1, int * AMR_LEVEL2, int * AMR_LEVEL3, int * AMR_COORD1, int * AMR_COORD2, int * AMR_COORD3, int nb1, int nb2, int nb3, int REF_1, int REF_2, int REF_3, float startx1, float startx2,
float startx3, float _dx1, float _dx2, float _dx3, int export_raytracing, int i_min, int i_max, int j_min, int j_max, int z_min, int z_max){
    int n,i,j,z,filesize, ii, gridsize_2D, index_2D, gridsize_3D, index_3D;
    int float_size = sizeof(double);
    int read_file;
    int P1, P2, P3, coord1, coord2, coord3, isize, jsize, i2, j2, z2;
    int dir, left, middle, right;
    int *level_array;
    float *x1_grad, *x2_grad, *x3_grad, *r_grad, *h_grad, *ph_grad, *gdet_grad, offset, offset1, offset2, offset3;
    char filename1[100],filename2[100];
    double trash[58];
    FILE *fin;

    isize=(z_max-z_min)*(j_max-j_min);
    jsize=(z_max-z_min);
    gridsize_2D=(i_max-i_min)*(j_max-j_min);
    gridsize_3D=(i_max-i_min)*(j_max-j_min)*(z_max-z_min);
    if(interpolate){
        //Allocate memory
        x1_grad=(float *)malloc(gridsize_3D*sizeof(float));
        x2_grad=(float *)malloc(gridsize_3D*sizeof(float));
        x3_grad=(float *)malloc(gridsize_3D*sizeof(float));
        r_grad=(float *)malloc(gridsize_3D*sizeof(float));
        h_grad=(float *)malloc(gridsize_3D*sizeof(float));
        ph_grad=(float *)malloc(gridsize_3D*sizeof(float));
        gdet_grad=(float *)malloc(gridsize_2D*sizeof(float));
    }
    level_array=(int *)calloc(gridsize_2D, sizeof(int));

    //Read in files and upscale to maximum AMR level
    #pragma omp parallel for private(n,i,j,z,offset1,offset2,offset3, filesize,ii, index_2D, index_3D, fin, filename1,filename2, trash,P1, P2, P3, coord1, coord2, coord3, i2, j2, z2, read_file)
    for(n=0;n<nb;n++){
        P1 = ACTIVE1 - AMR_LEVEL1[n_ord[n]];
        P2 = ACTIVE2 - AMR_LEVEL2[n_ord[n]];
        P3 = ACTIVE3 - AMR_LEVEL3[n_ord[n]];
        coord1 = AMR_COORD1[n_ord[n]];
        coord2 = AMR_COORD2[n_ord[n]];
        coord3 = AMR_COORD3[n_ord[n]];

        sprintf(filename2, "/gdumps/gdump%d", n_ord[n]);
        sprintf(filename1, directory);
        strcat(filename1,filename2);
        if( access(filename1, 0 ) != -1 ) {
            fin = fopen(filename1, "rb");
            fseek(fin, 0L, SEEK_END);
            filesize = ftell(fin);

            if(filesize>=(9*bs1*bs2*bs3+(bs1*bs2*49)*(axisym)+(bs1*bs2*bs3*49)*(!axisym))*float_size){
                fseek(fin, 0L, 0);
                read_file=0;
                offset1=(f1-((coord1)*((int)pow(1+REF_1,P1)*bs1))%f1)%f1;
                offset2=(f2-((coord2)*((int)pow(1+REF_2,P2)*bs2))%f2)%f2;
                offset3=(f3-((coord3)*((int)pow(1+REF_3,P3)*bs3))%f3)%f3;
                for(i2=(coord1)*((int)pow(1+REF_1,P1)*bs1)+offset1; i2<(coord1+1)*((int)pow(1+REF_1,P1)*bs1); i2+=(1*export_raytracing+f1*(1-export_raytracing)))for(j2=(coord2)*((int)pow(1+REF_2,P2)*bs2)+offset2; j2<(coord2+1)*((int)pow(1+REF_2,P2)*bs2); j2+=(1*export_raytracing+f2*(1-export_raytracing)))for(z2=(coord3)*((int)pow(1+REF_3,P3)*bs3)+offset3; z2<(coord3+1)*((int)pow(1+REF_3,P3)*bs3); z2+=(1*export_raytracing+f3*(1-export_raytracing))){
                //for(i2=(coord1)*((int)pow(1+REF_1,P1)*bs1)  +offset1; i2<(coord1+1)*((int)pow(1+REF_1,P1)*bs1); i2+=f1)for(j2=(coord2)*((int)pow(1+REF_2,P2)*bs2)+offset2; j2<(coord2+1)*((int)pow(1+REF_2,P2)*bs2); j2+=f2)for(z2=(coord3)*((int)pow(1+REF_3,P3)*bs3)+offset3; z2<(coord3+1)*((int)pow(1+REF_3,P3)*bs3); z2+=f3){
                    i=(i2-(coord1*((int)pow(1+REF_1,P1)*bs1)))/pow(1+REF_1, P1);
                    j=(j2-(coord2*((int)pow(1+REF_2,P2)*bs2)))/pow(1+REF_2, P2);
                    z=(z2-(coord3*((int)pow(1+REF_3,P3)*bs3)))/pow(1+REF_3, P3);

                    if(i2/f1>=i_min && i2/f1<i_max && j2/f2>=j_min && j2/f2<j_max && z2/f3>=z_min && z2/f3<z_max){
                        index_3D=(i2/f1-i_min)*isize+(j2/f2-j_min)*jsize+(z2/f3-z_min);
                        if(read_file==0) {
                            read_file=1;
                        }

                        fseek(fin, ((i)*bs2*bs3*9+(j)*bs3*9+(z)*9)*float_size, SEEK_SET);
                        fread(&trash[0], float_size, 9, fin);

                        if(export_raytracing){
                            float volume=(float)(f1*f2*f3);

                            x1[index_3D]+=trash[3]/volume;
                            x2[index_3D]+=trash[4]/volume;
                            x3[index_3D]+=trash[5]/volume;
                            r[index_3D]+=trash[6]/volume;
                            h[index_3D]+=trash[7]/volume;
                            ph[index_3D]+=trash[8]/volume;
                        }
                        else{
                            x1[index_3D]=trash[3];
                            x2[index_3D]=trash[4];
                            x3[index_3D]=trash[5];
                            r[index_3D]=trash[6];
                            h[index_3D]=trash[7];
                            ph[index_3D]=trash[8];
                        }
                    }
                }
                if(read_file==1){
                    if(axisym){
                        for(i2=(coord1)*((int)pow(1+REF_1,P1)*bs1)+offset1; i2<(coord1+1)*((int)pow(1+REF_1,P1)*bs1); i2+=f1)for(j2=(coord2)*((int)pow(1+REF_2,P2)*bs2)+offset2; j2<(coord2+1)*((int)pow(1+REF_2,P2)*bs2); j2+=f2){
                            i=(i2-(coord1*((int)pow(1+REF_1,P1)*bs1)))/pow(1+REF_1, P1);
                            j=(j2-(coord2*((int)pow(1+REF_2,P2)*bs2)))/pow(1+REF_2, P2);
                            i2/=f1;
                            j2/=f2;

                            if(i2>=i_min && i2<i_max && j2>=j_min && j2<j_max){
                                index_2D=(i2-i_min)*(j_max-j_min)+(j2-j_min);

                                if(MAX(AMR_LEVEL1[n_ord[n]],MAX(AMR_LEVEL2[n_ord[n]], AMR_LEVEL3[n_ord[n]]))>=level_array[index_2D]){
                                    fseek(fin, (9*bs1*bs2*bs3+(i)*bs2*49+(j)*49)*float_size, SEEK_SET);
                                    fread(&trash[0], float_size,49, fin);

                                    for(ii=0;ii<16;ii++) gcov[ii*gridsize_2D+index_2D]=trash[ii];
                                    for(ii=0;ii<16;ii++) gcon[ii*gridsize_2D+index_2D]=trash[16+ii];
                                    gdet[index_2D]=trash[32];
                                    for(ii=0;ii<16;ii++) dxdxp[ii*gridsize_2D+index_2D]=trash[32+1+ii];
                                    level_array[index_2D]=MAX(AMR_LEVEL1[n_ord[n]],MAX(AMR_LEVEL2[n_ord[n]], AMR_LEVEL3[n_ord[n]]));
                                }
                            }

                            i2*=f1;
                            j2*=f2;
                        }
                    }
                    else{
                        for(i2=(coord1)*((int)pow(1+REF_1,P1)*bs1)+offset1; i2<(coord1+1)*((int)pow(1+REF_1,P1)*bs1); i2+=f1)for(j2=(coord2)*((int)pow(1+REF_2,P2)*bs2)+offset2; j2<(coord2+1)*((int)pow(1+REF_2,P2)*bs2); j2+=f2)for(z2=(coord3)*((int)pow(1+REF_3,P3)*bs3)+offset3; z2<(coord3+1)*((int)pow(1+REF_3,P3)*bs3); z2+=f3){
                            i=(i2-(coord1*((int)pow(1+REF_1,P1)*bs1)))/pow(1+REF_1, P1);
                            j=(j2-(coord2*((int)pow(1+REF_2,P2)*bs2)))/pow(1+REF_2, P2);
                            z=(z2-(coord3*((int)pow(1+REF_3,P3)*bs3)))/pow(1+REF_3, P3);
                            i2/=f1;
                            j2/=f2;
                            z2/=f3;
                            if(i2>=i_min && i2<i_max && j2>=j_min && j2<j_max && z2>=z_min && z2<z_max){
                                index_3D=(i2-i_min)*isize+(j2-j_min)*jsize+(z2-z_min);

                                fseek(fin, (9*bs1*bs2*bs3+(i)*bs2*bs3*49+(j)*bs3*49+(z)*49)*float_size, SEEK_SET);
                                fread(&trash[0], float_size,49, fin);

                                for(ii=0;ii<16;ii++) gcov[ii*gridsize_3D+index_3D]=trash[ii];
                                for(ii=0;ii<16;ii++) gcon[ii*gridsize_3D+index_3D]=trash[16+ii];
                                gdet[index_3D]=trash[32];
                                for(ii=0;ii<16;ii++)dxdxp[ii*gridsize_3D+index_3D]=trash[32+1+ii];
                            }
                            i2*=f1;
                            j2*=f2;
                            z2*=f3;
                        }
                    }
                }
            }
	    else{
            	fprintf(stderr,"Possible data size error in file: %s \n", filename1);
	    }	
            fclose(fin);
        }
        else{
            fprintf(stderr,"Possible data corruption in file: %s \n", filename1);
        }
    }

    //Calculate gradient
     if(interpolate){
            for (dir=1;dir<=3;dir++){
                if(((dir==1) && ((f1 != 0) && ((f1 & (f1 - 1)) == 0))) || ((dir==2) && ((f2 != 0) && ((f2 & (f2 - 1)) == 0))) || ((dir==3) && ((f3 != 0) && ((f3 & (f3 - 1)) == 0)))){

                //Calculate gradient for lower level blocks
                #pragma omp parallel for private(n,i,j,z,offset,filesize,ii,index_2D, index_3D, fin, filename1,filename2, trash,P1, P2, P3, coord1, coord2, coord3, i2, j2, z2, read_file, left, middle, right, offset1, offset2, offset3)
                for(n=0;n<nb;n++){
                    P1 = ACTIVE1 - AMR_LEVEL1[n_ord[n]];
                    P2 = ACTIVE2 - AMR_LEVEL2[n_ord[n]];
                    P3 = ACTIVE3 - AMR_LEVEL3[n_ord[n]];
                    coord1 = AMR_COORD1[n_ord[n]];
                    coord2 = AMR_COORD2[n_ord[n]];
                    coord3 = AMR_COORD3[n_ord[n]];

                    offset1=(f1-((coord1)*((int)pow(1+REF_1,P1)*bs1))%f1)%f1;
                    offset2=(f2-((coord2)*((int)pow(1+REF_2,P2)*bs2))%f2)%f2;
                    offset3=(f3-((coord3)*((int)pow(1+REF_3,P3)*bs3))%f3)%f3;

                    for(i2=(coord1)*((int)pow(1+REF_1,P1)*bs1)+offset1; i2<(coord1+1)*((int)pow(1+REF_1,P1)*bs1); i2+=f1)for(j2=(coord2)*((int)pow(1+REF_2,P2)*bs2)+offset2; j2<(coord2+1)*((int)pow(1+REF_2,P2)*bs2); j2+=f2)for(z2=(coord3)*((int)pow(1+REF_3,P3)*bs3)+offset3; z2<(coord3+1)*((int)pow(1+REF_3,P3)*bs3); z2+=f3){
                        i2/=f1;
                        j2/=f2;
                        z2/=f3;

                        if(i2>=i_min && i2<i_max && j2>=j_min && j2<j_max && z2>=z_min && z2<z_max){
                            index_3D=(i2-i_min)*isize+(j2-j_min)*jsize+(z2-z_min);
                            index_2D=(i2-i_min)*(j_max-j_min)+(j2-j_min);
                            read_file=1;

                            if(dir==1){
                                left=index_3D-(i2-i_min)*isize+isize*MAX(0, (i2-i_min)-MAX(ceil(pow(1+REF_1, P1)/f1),1));
                                middle=index_3D;
                                right=index_3D-(i2-i_min)*isize+isize*MIN((i_max-i_min)-1, (i2-i_min)+MAX(ceil(pow(1+REF_1, P1)/f1),1));
                            }
                            else if(dir==2){
                                left=index_3D-(j2-j_min)*jsize+jsize*MAX(0, (j2-j_min)-MAX(ceil(pow(1+REF_2, P2)/f2),1));
                                middle=index_3D;
                                right=index_3D-(j2-j_min)*jsize+jsize*MIN((j_max-j_min)-1, (j2-j_min)+MAX(ceil(pow(1+REF_2, P2)/f2),1));
                            }
                            else {
                                left=index_3D-(z2-z_min)+MAX(0, (z2-z_min)-MAX(ceil(pow(1+REF_3, P3)/f3),1));
                                middle=index_3D;
                                right=index_3D-(z2-z_min)+MIN((z_max-z_min)-1, (z2-z_min)+MAX(ceil(pow(1+REF_3, P3)/f3),1));
                            }

                            float x_left, x_middle, x_right;
                            if(dir==1){
                                x_left=x1[left];
                                x_middle=x1[middle];
                                x_right=x1[right];
                            }
                            else if(dir==2){
                                x_left=x2[left];
                                x_middle=x2[middle];
                                x_right=x2[right];
                            }
                            else{
                                x_left=x3[left];
                                x_middle=x3[middle];
                                x_right=x3[right];
                            }
                            x1_grad[index_3D]=x2_grad[index_3D]=x3_grad[index_3D]=r_grad[index_3D]=h_grad[index_3D]=ph_grad[index_3D]=0.0;
                            //x1_grad[index_3D]=slope_lim(x1[left],x1[middle],x1[right]);
                            //x2_grad[index_3D]=slope_lim(x2[left],x2[middle],x2[right]);
                            //x3_grad[index_3D]=slope_lim(x3[left],x3[middle],x3[right]);
                            r_grad[index_3D]=slope_lim(r[left],r[middle],r[right], x_left, x_middle, x_right);
                            h_grad[index_3D]=slope_lim(h[left],h[middle],h[right], x_left, x_middle, x_right);
                            ph_grad[index_3D]=slope_lim(ph[left],ph[middle],ph[right], x_left, x_middle, x_right);

                            if(z2==z_min && axisym==1 && (dir==1 || dir==2)){
                                 if(dir==1){
                                    left=index_2D-(i2-i_min)*(j_max-j_min)+(j_max-j_min)*MAX(0, (i2-i_min)-MAX(ceil(pow(1+REF_1, P1)/f1),1));
                                    middle=index_2D;
                                    right=index_2D-(i2-i_min)*(j_max-j_min)+(j_max-j_min)*MIN((i_max-i_min)-1, (i2-i_min)+MAX(ceil(pow(1+REF_1, P1)/f1),1));
                                }
                                else if(dir==2){
                                    left=index_2D-(j2-j_min)+MAX(0, (j2-j_min)-MAX(ceil(pow(1+REF_2, P2)/f2),1));
                                    middle=index_2D;
                                    right=index_2D-(j2-j_min)+MIN((j_max-j_min)-1, (j2-j_min)+MAX(ceil(pow(1+REF_2, P2)/f2),1));
                                }
                                gdet_grad[index_2D]=0.0;
                                gdet_grad[index_2D]=slope_lim(gdet[left],gdet[middle],gdet[right], x_left, x_middle, x_right);
                            }
                        }

                        i2*=f1;
                        j2*=f2;
                        z2*=f3;
                    }
                }

                //Add gradient
                #pragma omp parallel for private(n,i,j,z,offset,filesize,ii,index_2D, index_3D, fin, filename1,filename2, trash,P1, P2, P3, coord1, coord2, coord3,  i2, j2, z2, read_file, left, middle, right, offset1, offset2, offset3)
                for(n=0;n<nb;n++){
                    P1 = ACTIVE1 - AMR_LEVEL1[n_ord[n]];
                    P2 = ACTIVE2 - AMR_LEVEL2[n_ord[n]];
                    P3 = ACTIVE3 - AMR_LEVEL3[n_ord[n]];
                    coord1 = AMR_COORD1[n_ord[n]];
                    coord2 = AMR_COORD2[n_ord[n]];
                    coord3 = AMR_COORD3[n_ord[n]];

                    offset1=(f1-((coord1)*((int)pow(1+REF_1,P1)*bs1))%f1)%f1;
                    offset2=(f2-((coord2)*((int)pow(1+REF_2,P2)*bs2))%f2)%f2;
                    offset3=(f3-((coord3)*((int)pow(1+REF_3,P3)*bs3))%f3)%f3;

                    for(i2=(coord1)*((int)pow(1+REF_1,P1)*bs1)+offset1; i2<(coord1+1)*((int)pow(1+REF_1,P1)*bs1); i2+=f1)for(j2=(coord2)*((int)pow(1+REF_2,P2)*bs2)+offset2; j2<(coord2+1)*((int)pow(1+REF_2,P2)*bs2); j2+=f2)for(z2=(coord3)*((int)pow(1+REF_3,P3)*bs3)+offset3; z2<(coord3+1)*((int)pow(1+REF_3,P3)*bs3); z2+=f3){
                        i2/=f1;
                        j2/=f2;
                        z2/=f3;

                        if(i2>=i_min && i2<i_max && j2>=j_min && j2<j_max && z2>=z_min && z2<z_max){
                            index_3D=(i2-i_min)*isize+(j2-j_min)*jsize+(z2-z_min);
                            index_2D=(i2-i_min)*(j_max-j_min)+(j2-j_min);
                            read_file=1;
                            if(dir==1){
                                offset=-(x1[index_3D]-(startx1+(i2+0.5)*_dx1));
                                //offset = ((0.5 + Li)-0.5*pow(1+REF_1, P1))/f1;
                            }
                            else if(dir==2){
                                offset=-(x2[index_3D]-(startx2+(j2+0.5)*_dx2));
                                //offset = ((0.5 + Lj)-0.5*pow(1+REF_2, P2))/f2;
                            }
                            else {
                                offset=-(x3[index_3D]-(startx3+(z2+0.5)*_dx3));
                                //offset = ((0.5 + Lz)-0.5*pow(1+REF_3, P3))/f3;
                            }
                            //x1[index_3D]=x1[index_3D]+offset*x1_grad[index_3D];
                            //x2[index_3D]=x2[index_3D]+offset*x2_grad[index_3D];
                            //x3[index_3D]=x3[index_3D]+offset*x3_grad[index_3D];
                            r[index_3D]=r[index_3D]+offset*r_grad[index_3D];
                            h[index_3D]=h[index_3D]+offset*h_grad[index_3D];
                            ph[index_3D]=ph[index_3D]+offset*ph_grad[index_3D];
                            if(z2==z_min && axisym==1 && (dir==1 || dir==2)){
                                gdet[index_2D]=gdet[index_2D]+offset*gdet_grad[index_2D];
                            }
                        }

                        i2*=f1;
                        j2*=f2;
                        z2*=f3;
                    }
                }
            }
        }
    }
    if(interpolate){
        free(x1_grad);
        free(x2_grad);
        free(x3_grad);
        free(r_grad);
        free(h_grad);
        free(ph_grad);
        free(gdet_grad);
    }
    free(level_array);
}

void kernel_rgdump_write(int flag, char *dir, int axisym, int *n_ord, int f1, int f2, int f3, int nb, int bs1, int bs2, int bs3, float* x1,float* x2,float* x3,float* r,float* h,float* ph,float* gcov,float* gcon,float* dxdxp,float* gdet){
    int n,i,j,z,k, ii, gridsize_2D, index_2D, gridsize_3D, index_3D;
    int float_size = sizeof(double);
    int bs1new=bs1/f1;
    int bs2new=bs2/f2;
    int bs3new=bs3/f3;
    char filename1[100],filename2[100];
    double trash[58];
    FILE *fin;
    double *temp_array;

    //#pragma omp parallel for private(n,i,j,z,k, ii,gridsize_2D, index_2D, gridsize_3D, index_3D, fin, filename1,filename2, trash,temp_array)
    for(n=0;n<nb;n++){
        temp_array=(double *)calloc(1*(58*bs1new*bs2new*bs3new), sizeof(double));
        sprintf(filename2, "/gdumps/gdump%d", n_ord[n]);
        sprintf(filename1, dir);
        strcat(filename1,filename2);
        if( access(filename1, 0 ) == -1 ) {
            fin = fopen(filename1, "wb");
            for(i=0;i<bs1new;i++)for(j=0;j<bs2new;j++)for(z=0;z<bs3new;z++){
                index_3D=n*bs1new*bs2new*bs3new+i*bs3new*bs2new+j*bs3new+z;
                gridsize_3D=nb*bs1new*bs2new*bs3new;
                trash[0]=0;
                trash[1]=0;
                trash[2]=0;
                trash[3]=x1[index_3D];
                trash[4]=x2[index_3D];
                trash[5]=x3[index_3D];
                trash[6]=r[index_3D];
                trash[7]=h[index_3D];
                trash[8]=ph[index_3D];
                for(k=0;k<9;k++)temp_array[0*58*bs1new*bs2new*bs3new+i*bs2new*bs3new*9+j*bs3new*9+z*9+k]=trash[k];
            }
            fwrite(&temp_array[0*58*bs1new*bs2new*bs3new], float_size,9*bs1new*bs2new*bs3new, fin);

            if(axisym){
                for(i=0;i<bs1new;i++)for(j=0;j<bs2new;j++){
                    index_2D=n*bs1new*bs2new+i*bs2new+j;
                    gridsize_2D=nb*bs1new*bs2new;
                    for(ii=0;ii<16;ii++) trash[ii]=gcov[ii*gridsize_2D+index_2D];
                    for(ii=0;ii<16;ii++) trash[16+ii]=gcon[ii*gridsize_2D+index_2D];
                    trash[32]=gdet[index_2D];
                    for(ii=0;ii<16;ii++) trash[32+1+ii]=dxdxp[ii*gridsize_2D+index_2D];
                    for(k=0;k<49;k++)temp_array[0*58*bs1new*bs2new*bs3new+i*bs2new*49+j*49+k]=trash[k];
                }
                fwrite(&temp_array[0*58*bs1new*bs2new*bs3new], float_size,49*bs1new*bs2new, fin);
            }
            else{
                for(i=0;i<bs1new;i++)for(j=0;j<bs2new;j++)for(z=0;z<bs3new;z++){
                    index_3D=n*bs1new*bs2new*bs3new+i*bs3new*bs2new+j*bs3new+z;
                    gridsize_3D=nb*bs1new*bs2new*bs3new;
                    for(ii=0;ii<16;ii++) trash[ii]=gcov[ii*gridsize_3D+index_3D];
                    for(ii=0;ii<16;ii++) trash[16+ii]=gcon[ii*gridsize_3D+index_3D];
                    trash[32]=gdet[index_3D];
                    for(ii=0;ii<16;ii++)trash[32+1+ii]=dxdxp[ii*gridsize_3D+index_3D];
                    for(k=0;k<49;k++)temp_array[0*58*bs1new*bs2new*bs3new+i*bs2new*bs3new*49+j*bs3new*49+z*49+k]=trash[k];
                }
                fwrite(&temp_array[0*58*bs1new*bs2new*bs3new], float_size,49*bs1new*bs2new*bs3new, fin);
            }
            fclose(fin);
        }
        free(temp_array);
    }
}

void kernel_rdump_new(int flag, int RAD_M1, int RESISTIVE, int TWO_T, int P_NUM,  char *directory, int dump, int n_active_total, int f1,int f2,int f3, int nb, int bs1, int bs2, int bs3, float* rho,float* ug, float* uu, float* B,float* E,float* E_rad,float* uu_rad,float* TE,float* TI,float* photon_number,float* gcov,float* gcon, int axisym){
    #pragma omp parallel
	{
        int n,i,j,z, ii,u, gridsize_3D, index_3D, n_start,n_end, filesize;
        int float_size = sizeof(float);
        int bs1new=bs1/f1;
        int bs2new=bs2/f2;
        int bs3new=bs3/f3;
        int keep_looping, num_threads, thread_id;
        int NPR=9+5*RAD_M1+3*RESISTIVE+2*TWO_T+1*P_NUM;
        char filename1[100],filename2[100];
        float trash[14];
        FILE *fin;

        keep_looping=1;
        u=0;
        n=0;
        num_threads = omp_get_num_threads();
		thread_id = omp_get_thread_num();

        while(keep_looping){
            if(flag)sprintf(filename2, "/dumps%d/new_dump", dump);
            else sprintf(filename2, "/dumps%d/new_dump%d", dump, u);
            sprintf(filename1, directory);
            strcat(filename1,filename2);

            if( access(filename1, 0 ) != -1 ) {
                fin = fopen(filename1, "rb");
                fseek(fin, 0L, SEEK_END);
                filesize = ftell(fin);
                if(filesize%(NPR*bs1*bs2*bs3*float_size)==0){
                    fseek(fin, 0L, 0);
                    n_start=n;
                    n_end=n+filesize/(NPR*bs1*bs2*bs3*float_size);
                    for(n=n_start; n<n_end; n++){
                        if(u%num_threads==thread_id){
                            for(i=0;i<bs1new;i++)for(j=0;j<bs2new;j++)for(z=0;z<bs3new;z++){
                                index_3D=n*bs1new*bs2new*bs3new+i*bs3new*bs2new+j*bs3new+z;
                                gridsize_3D=nb*bs1new*bs2new*bs3new;

                                fseek(fin, ((n-n_start)*NPR*bs1*bs2*bs3+(i*f1)*bs2*bs3*NPR+(j*f2)*bs3*NPR+(z*f3)*NPR)*float_size, SEEK_SET);
                                fread(&trash[0], float_size, NPR, fin);

                                rho[index_3D]=trash[0];
                                ug[index_3D]=trash[1];
                                for(ii=0;ii<4;ii++)uu[(ii)*gridsize_3D+index_3D]=trash[2+ii];

                                //Radiation variables
                                if(RAD_M1==1){
                                    E_rad[index_3D]=trash[9];
                                    for(ii=0;ii<4;ii++) uu_rad[(ii)*gridsize_3D+index_3D]=trash[10+ii];
                                }
                                 if(RESISTIVE==1){
                                    for(ii=0;ii<3;ii++)E[(ii+1)*gridsize_3D+index_3D]=trash[9+5*RAD_M1+ii];
                                }
                                if(TWO_T==1){
                                    TE[index_3D]=trash[9+5*RAD_M1+3*RESISTIVE];
                                    TI[index_3D]=trash[9+5*RAD_M1+3*RESISTIVE+1];
                                }
                                if(P_NUM==1){
                                    photon_number[index_3D]=trash[9+5*RAD_M1+3*RESISTIVE+2*TWO_T];
                                }
                            }
                        }
                        if(n==n_active_total-1){
                            keep_looping=0;
                            break;
                        }
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

        if(n!=n_active_total-1 && thread_id==0){
            fprintf(stderr,"Possible data corruption in file: %s \n", filename1);
        }
    }
}

 float misc_source(float rho, float ug, float bsq, float ucon[4], float r, float ucov[4], float H_OVER_R, float a, float gam){
	double epsilon = ug / rho;
	double om_kepler = 1. / (pow(r, 3. / 2.) + a);
	double T_target = 3.141592 / 2.*pow(H_OVER_R*r*om_kepler, 2.);
	double Y = (gam - 1.)*epsilon / T_target;
	double lambda = om_kepler*ug * sqrt(Y - 1. + fabs(Y - 1.));
	if (bsq / rho<1. || r<10.){
		return ((float)(-ucov[0] * lambda));
	}
	else{
		return 0.0;
	}
}

void kernel_rdump_griddata(int flag, int interpolate, int RAD_M1, int RESISTIVE, int TWO_T, int P_NUM, char *directory, int dump, int n_active_total, int f1,int f2,int f3, int nb, int bs1, int bs2, int bs3, float* rho,float* ug, float* uu, float* B,float* E,float* E_rad,float* uu_rad,float* TE,float* TI,float* photon_number,float* gcov,float* gcon,
int axisym,int* n_ord,int ACTIVE1, int ACTIVE2, int ACTIVE3, int * AMR_LEVEL1, int * AMR_LEVEL2, int * AMR_LEVEL3, int * AMR_COORD1, int * AMR_COORD2, int * AMR_COORD3, int nb1, int nb2, int nb3, int REF_1, int REF_2, int REF_3, int export_raytracing,float DISK_THICKNESS,
float a, float gam, float* Rdot, float* bsq, float* r, float startx1, float startx2, float startx3, float _dx1, float _dx2, float _dx3, float* x1, float* x2, float* x3, int i_min, int i_max, int j_min, int j_max, int z_min, int z_max){
    float *rho_grad, *ug_grad, *uu_grad, *B_grad,*E_grad, *E_rad_grad, *uu_rad_grad, *Rdot_grad, *TE_grad, *TI_grad, *photon_number_grad;
    int float_size = sizeof(float);
    int gridsize_3D;

    //Allocate memory
    gridsize_3D=(i_max-i_min)*(j_max-j_min)*(z_max-z_min);
    if(interpolate){
        rho_grad=(float *)malloc(gridsize_3D*sizeof(float));
        ug_grad=(float *)malloc(gridsize_3D*sizeof(float));
        uu_grad=(float *)malloc(4*gridsize_3D*sizeof(float));
        B_grad=(float *)malloc(4*gridsize_3D*sizeof(float));
        if(export_raytracing) Rdot_grad= (float *)malloc(gridsize_3D*sizeof(float));
        if(RAD_M1){
            E_rad_grad=(float *)malloc(gridsize_3D*sizeof(float));
            uu_rad_grad=(float *)malloc(4*gridsize_3D*sizeof(float));
        }
        if(RESISTIVE){
            E_grad=(float *)malloc(4*gridsize_3D*sizeof(float));
        }
        if(TWO_T){
            TE_grad=(float *)malloc(gridsize_3D*sizeof(float));
            TI_grad=(float *)malloc(gridsize_3D*sizeof(float));
        }
        if(P_NUM){
            photon_number_grad=(float *)malloc(gridsize_3D*sizeof(float));
        }
    }

	#pragma omp parallel
	{
	    int n,i,j,z, ii,u, index_3D, gridsize_2D, index_2D, n_start,n_end, filesize;
	    int dir, left, middle, right, offset1, offset2, offset3;
        float uu_local[4],ud_local[4],B_local[4],bu_local[4],bd_local[4], gcov_local[16], bsq;

        int keep_looping, num_threads, thread_id, read_file;
        int P1, P2, P3, coord1, coord2, coord3, isize, jsize, i2, j2, z2;
        int NPR=9+5*RAD_M1+3*RESISTIVE+2*TWO_T+P_NUM;
        char filename1[400],filename2[200];
        float trash[14], gamma, alpha,beta[4], vkerr[4], offset;
        FILE *fin;
        isize=(j_max-j_min)*(z_max-z_min);
        jsize=(z_max-z_min);
        keep_looping=1;
        u=0;
        n=0;
	    num_threads = omp_get_num_threads();
	    thread_id = omp_get_thread_num();

        while(keep_looping){
        	if(flag) sprintf(filename2, "/dumps%d/new_dump", dump);
            else sprintf(filename2, "/dumps%d/new_dump%d", dump, u);
            sprintf(filename1, directory);
            strcat(filename1,filename2);
            if(access(filename1, 0 ) != -1 ) {
                fin = fopen(filename1, "rb");
                fseek(fin, 0L, SEEK_END);
                filesize = ftell(fin);
                if(filesize%(NPR*bs1*bs2*bs3*float_size)==0){
                    fseek(fin, 0L, 0);
                    n_start=n;
                    n_end=n+filesize/(NPR*bs1*bs2*bs3*float_size);
                    for(n=n_start; n<n_end; n++){
                        if(u%num_threads==thread_id){
                            P1 = ACTIVE1 - AMR_LEVEL1[n_ord[n]];
                            P2 = ACTIVE2 - AMR_LEVEL2[n_ord[n]];
                            P3 = ACTIVE3 - AMR_LEVEL3[n_ord[n]];
                            coord1 = AMR_COORD1[n_ord[n]];
                            coord2 = AMR_COORD2[n_ord[n]];
                            coord3 = AMR_COORD3[n_ord[n]];
                            read_file=0;

                            offset1=(f1-((coord1)*((int)pow(1+REF_1,P1)*bs1))%f1)%f1;
                            offset2=(f2-((coord2)*((int)pow(1+REF_2,P2)*bs2))%f2)%f2;
                            offset3=(f3-((coord3)*((int)pow(1+REF_3,P3)*bs3))%f3)%f3;

                            for(i2=(coord1)*((int)pow(1+REF_1,P1)*bs1)+offset1; i2<(coord1+1)*((int)pow(1+REF_1,P1)*bs1); i2+=(1*export_raytracing+f1*(1-export_raytracing)))for(j2=(coord2)*((int)pow(1+REF_2,P2)*bs2)+offset2; j2<(coord2+1)*((int)pow(1+REF_2,P2)*bs2); j2+=(1*export_raytracing+f2*(1-export_raytracing)))for(z2=(coord3)*((int)pow(1+REF_3,P3)*bs3)+offset3; z2<(coord3+1)*((int)pow(1+REF_3,P3)*bs3); z2+=(1*export_raytracing+f3*(1-export_raytracing))){
                                i=(i2-(coord1*((int)pow(1+REF_1,P1)*bs1)))/pow(1+REF_1, P1);
                                j=(j2-(coord2*((int)pow(1+REF_2,P2)*bs2)))/pow(1+REF_2, P2);
                                z=(z2-(coord3*((int)pow(1+REF_3,P3)*bs3)))/pow(1+REF_3, P3);

                                if(i2/f1>=i_min && i2/f1<i_max && j2/f2>=j_min && j2/f2<j_max && z2/f3>=z_min && z2/f3<z_max){
                                    index_3D=(i2/f1-i_min)*isize+(j2/f2-j_min)*jsize+(z2/f3-z_min);
                                    if(axisym){
                                        index_2D=(i2/f1-i_min)*(j_max-j_min)+(j2/f2-j_min);
                                        gridsize_2D=(i_max-i_min)*(j_max-j_min);
                                    }
                                    else{
                                        index_2D=index_3D;
                                        gridsize_2D=gridsize_3D;
                                    }
                                    if(read_file==0) read_file=1;

                                    fseek(fin, ((n-n_start)*NPR*bs1*bs2*bs3+(i)*bs2*bs3*NPR+(j)*bs3*NPR+(z)*NPR)*float_size, SEEK_SET);
                                    fread(&trash[0], float_size, NPR, fin);

                                    if(export_raytracing==1){
                                        float volume=(float)(f1*f2*f3);

                                        rho[index_3D]+=trash[0]/volume;
                                        ug[index_3D]+=trash[1]/volume;
                                        for(ii=0;ii<3;ii++) B[(ii+1)*gridsize_3D+index_3D]+=trash[6+ii]/volume;

                                        //Do velocity conversion
                                        alpha= 1. / sqrt(-gcon[index_2D]);
                                        for(ii=1;ii<4;ii++) beta[ii]=gcon[ii*gridsize_2D+index_2D] * alpha * alpha;
                                        gamma=trash[2]*alpha;
                                        for(ii=1;ii<4;ii++)uu[(ii)*gridsize_3D+index_3D]+=(gamma*beta[ii]/alpha + trash[2+ii])/volume;

                                        //Calculate source term
                                        for (ii=0; ii<4; ii++) {
                                            uu_local[ii]= trash[2+ii];
                                            bu_local[ii]= 0.0;
                                        }
                                        for (ii=0; ii<3; ii++)B_local[ii+1]= trash[6+ii];
                                        for (ii=0; ii<16; ii++) gcov_local[ii]= gcov[ii*gridsize_2D+index_2D];
                                        bu_calc(B_local, uu_local, bu_local, gcov_local);
                                        lower_c(bu_local, bd_local, gcov_local);
                                        lower_c(uu_local, ud_local, gcov_local);
                                        bsq = bu_local[0] * bd_local[0] + bu_local[1] * bd_local[1] + bu_local[2] * bd_local[2] + bu_local[3] * bd_local[3];
                                        Rdot[index_3D]+=misc_source(trash[0], trash[1], bsq, uu_local, r[index_3D], ud_local, DISK_THICKNESS, a, gam)/volume;

                                        if(RAD_M1){
                                            for(ii=1;ii<4;ii++)vkerr[ii]=0.;
                                            E_rad[index_3D]+=trash[9]/volume;
                                            gamma=trash[10]*alpha;
                                            for(ii=1;ii<4;ii++) uu_rad[(ii)*gridsize_3D+index_3D]+=(gamma*beta[ii]/alpha + trash[10+ii])/volume;
                                        }
                                        if(RESISTIVE){
                                            for(ii=0;ii<3;ii++) E[(ii+1)*gridsize_3D+index_3D]+=(trash[9+5*RAD_M1+ii])/volume;
                                        }
                                        if(TWO_T){
                                            TE[index_3D]+=trash[9+5*RAD_M1+3*RESISTIVE]/volume;
                                            TI[index_3D]+=trash[9+5*RAD_M1+3*RESISTIVE+1]/volume;
                                        }
                                        if(P_NUM){
                                            photon_number[index_3D]+=trash[9+5*RAD_M1+3*RESISTIVE+2*TWO_T]/volume;
                                        }
                                    }
                                    else{
                                        rho[index_3D]=trash[0];
                                        ug[index_3D]=trash[1];
                                        for(ii=0;ii<3;ii++) B[(ii+1)*gridsize_3D+index_3D]=trash[6+ii];

                                        //Do velocity conversion
                                       if(interpolate){
                                            alpha= 1. / sqrt(-gcon[index_2D]);
                                            for(ii=1;ii<4;ii++) beta[ii]=gcon[ii*gridsize_2D+index_2D] * alpha * alpha;
                                            gamma=trash[2]*alpha;
                                            for(ii=1;ii<4;ii++)uu[(ii)*gridsize_3D+index_3D]=(gamma*beta[ii]/alpha + trash[2+ii]) ;
                                       }
                                       else{
                                            for(ii=0;ii<4;ii++)uu[(ii)*gridsize_3D+index_3D]=(trash[2+ii]) ;
                                       }
                                        //Radiation variables
                                        if(RAD_M1){
                                            E_rad[index_3D]=trash[9];
                                            if(interpolate){
                                                for(ii=1;ii<4;ii++)vkerr[ii]=0.;
                                                gamma=trash[10]*alpha;
                                                for(ii=0;ii<4;ii++) uu_rad[(ii)*gridsize_3D+index_3D]=(gamma*beta[ii]/alpha + trash[10+ii]) ;
                                            }
                                            else{
                                                for(ii=0;ii<4;ii++) uu_rad[(ii)*gridsize_3D+index_3D]=(trash[10+ii]) ;
                                            }
                                        }
                                        if(RESISTIVE){
                                            for(ii=0;ii<3;ii++) E[(ii+1)*gridsize_3D+index_3D]=(trash[9+5*RAD_M1+ii]);
                                        }
                                        if(TWO_T){
                                            TE[index_3D]=trash[9+5*RAD_M1+3*RESISTIVE];
                                            TI[index_3D]=trash[9+5*RAD_M1+3*RESISTIVE+1];
                                        }
                                        if(P_NUM){
                                            photon_number[index_3D]=trash[9+5*RAD_M1+3*RESISTIVE+2*TWO_T];
                                        }
                                    }
                                }
                            }
                        }
                        if(n==n_active_total-1){
                            keep_looping=0;
                            break;
                        }
                    }
                }
                else{
                    fprintf(stderr,"Possible data corruption (wrong size) in file: %s \n", filename1);
                }
                fclose(fin);
            }
            else keep_looping=0;
            u++;
        }

        //Check for data corruption
        if(n!=n_active_total-1 && thread_id==0){
            fprintf(stderr,"Possible data corruption in file: %s \n", filename1);
        }

        if(interpolate){
            for (dir=1;dir<=3;dir++){
                if(((dir==1) && ((f1 != 0) && ((f1 & (f1 - 1)) == 0))) || ((dir==2) && ((f2 != 0) && ((f2 & (f2 - 1)) == 0))) || ((dir==3) && ((f3 != 0) && ((f3 & (f3 - 1)) == 0)))){
                    //Calculate gradient for lower level blocks
                    //#pragma omp for private(n,i,j,z,offset,ii, index_3D, trash,P1, P2, P3, coord1, coord2, coord3, i2, j2, z2, read_file, left, middle, right, offset1, offset2, offset3)
                    for(n=0;n<n_active_total;n++){
						if(n%num_threads==thread_id){
							P1 = ACTIVE1 - AMR_LEVEL1[n_ord[n]];
							P2 = ACTIVE2 - AMR_LEVEL2[n_ord[n]];
							P3 = ACTIVE3 - AMR_LEVEL3[n_ord[n]];
							coord1 = AMR_COORD1[n_ord[n]];
							coord2 = AMR_COORD2[n_ord[n]];
							coord3 = AMR_COORD3[n_ord[n]];

							offset1=(f1-((coord1)*((int)pow(1+REF_1,P1)*bs1))%f1)%f1;
							offset2=(f2-((coord2)*((int)pow(1+REF_2,P2)*bs2))%f2)%f2;
							offset3=(f3-((coord3)*((int)pow(1+REF_3,P3)*bs3))%f3)%f3;

							for(i2=(coord1)*((int)pow(1+REF_1,P1)*bs1)+offset1; i2<(coord1+1)*((int)pow(1+REF_1,P1)*bs1); i2+=f1)for(j2=(coord2)*((int)pow(1+REF_2,P2)*bs2)+offset2; j2<(coord2+1)*((int)pow(1+REF_2,P2)*bs2); j2+=f2)for(z2=(coord3)*((int)pow(1+REF_3,P3)*bs3)+offset3; z2<(coord3+1)*((int)pow(1+REF_3,P3)*bs3); z2+=f3){
								i2/=f1;
								j2/=f2;
								z2/=f3;

								if(i2>=i_min && i2<i_max && j2>=j_min && j2<j_max && z2>=z_min && z2<z_max){
									index_3D=(i2-i_min)*isize+(j2-j_min)*jsize+(z2-z_min);
									read_file=1;

									if(dir==1){
										left=index_3D-(i2-i_min)*isize+isize*MAX(0, (i2-i_min)-MAX(ceil(pow(1+REF_1, P1)/f1),1));
										middle=index_3D;
										right=index_3D-(i2-i_min)*isize+isize*MIN((i_max-i_min)-1, (i2-i_min)+MAX(ceil(pow(1+REF_1, P1)/f1),1));
									}
									else if(dir==2){
										left=index_3D-(j2-j_min)*jsize+jsize*MAX(0, (j2-j_min)-MAX(ceil(pow(1+REF_2, P2)/f2),1));
										middle=index_3D;
										right=index_3D-(j2-j_min)*jsize+jsize*MIN((j_max-j_min)-1, (j2-j_min)+MAX(ceil(pow(1+REF_2, P2)/f2),1));
									}
									else {
										left=index_3D-(z2-z_min)+MAX(0, (z2-z_min)-MAX(ceil(pow(1+REF_3, P3)/f3),1));
										middle=index_3D;
										right=index_3D-(z2-z_min)+MIN((z_max-z_min)-1, (z2-z_min)+MAX(ceil(pow(1+REF_3, P3)/f3),1));
									}

									float x_left, x_middle, x_right;
									if(dir==1){
										x_left=x1[left];
										x_middle=x1[middle];
										x_right=x1[right];
									}
									else if(dir==2){
										x_left=x2[left];
										x_middle=x2[middle];
										x_right=x2[right];
									}
									else{
										x_left=x3[left];
										x_middle=x3[middle];
										x_right=x3[right];
									}

									//Reset gradients to zero
									rho_grad[index_3D]= ug_grad[index_3D]=0.0;
									for(ii=0; ii<4; ii++)uu_grad[ii*gridsize_3D+index_3D]=0.0;
									for(ii=0; ii<4; ii++)B_grad[ii*gridsize_3D+index_3D]=0.0;
									if(export_raytracing){
										Rdot_grad[index_3D]=0.;
									}
                                    if(fabs(x_left-x_middle)>0.000001 && fabs(x_right-x_middle)>0.000001){
                                        rho_grad[index_3D]=slope_lim(rho[left],rho[middle],rho[right], x_left, x_middle, x_right);
                                        ug_grad[index_3D]=slope_lim(ug[left],ug[middle],ug[right], x_left, x_middle, x_right);
                                        for(ii=0; ii<4; ii++)uu_grad[ii*gridsize_3D+index_3D]=slope_lim(uu[ii*gridsize_3D+left],uu[ii*gridsize_3D+middle],uu[ii*gridsize_3D+right], x_left, x_middle, x_right);
                                        for(ii=0; ii<4; ii++)B_grad[ii*gridsize_3D+index_3D]=slope_lim(B[ii*gridsize_3D+left],B[ii*gridsize_3D+middle],B[ii*gridsize_3D+right], x_left, x_middle, x_right);
                                        if(export_raytracing){
                                            Rdot_grad[index_3D]=slope_lim(Rdot[left],Rdot[middle],Rdot[right], x_left, x_middle, x_right);
                                        }

                                        //Radiation variables
                                        if(RAD_M1){
                                            E_rad_grad[index_3D]=0.0;
                                            for(ii=0; ii<4; ii++)uu_rad_grad[ii*gridsize_3D+index_3D]=0.0;
                                            if(fabs(left-middle)>pow(10.,-20) && fabs(right-middle)>pow(10.,-20)){
                                                E_rad_grad[index_3D]=slope_lim(E_rad[left],E_rad[middle],E_rad[right], x_left, x_middle, x_right);
                                                for(ii=0; ii<4; ii++)uu_rad_grad[ii*gridsize_3D+index_3D]=slope_lim(uu_rad[ii*gridsize_3D+left],uu_rad[ii*gridsize_3D+middle],uu_rad[ii*gridsize_3D+right], x_left, x_middle, x_right);
                                            }
                                        }

                                        //Resistive variables
                                        if(RESISTIVE){
                                            for(ii=0; ii<4; ii++)E_grad[ii*gridsize_3D+index_3D]=0.0;
                                            if(fabs(left-middle)>pow(10.,-20) && fabs(right-middle)>pow(10.,-20)){
                                                for(ii=0; ii<4; ii++)E_grad[ii*gridsize_3D+index_3D]=slope_lim(E[ii*gridsize_3D+left],E[ii*gridsize_3D+middle],E[ii*gridsize_3D+right], x_left, x_middle, x_right);
                                            }
                                        }
                                        if(TWO_T){
                                            TE_grad[index_3D]=0.0;
                                            if(fabs(left-middle)>pow(10.,-20) && fabs(right-middle)>pow(10.,-20)){
                                                TE_grad[index_3D]=slope_lim(TE[left],TE[middle],TE[right], x_left, x_middle, x_right);
                                            }
                                            TI_grad[index_3D]=0.0;
                                            if(fabs(left-middle)>pow(10.,-20) && fabs(right-middle)>pow(10.,-20)){
                                                TI_grad[index_3D]=slope_lim(TI[left],TI[middle],TI[right], x_left, x_middle, x_right);
                                            }
                                        }
                                        if(P_NUM){
                                            photon_number_grad[index_3D]=0.0;
                                            if(fabs(left-middle)>pow(10.,-20) && fabs(right-middle)>pow(10.,-20)){
                                                photon_number_grad[index_3D]=slope_lim(photon_number[left],photon_number[middle],photon_number[right], x_left, x_middle, x_right);
                                            }
                                        }
									}
								}

								i2*=f1;
								j2*=f2;
								z2*=f3;
							}
					    }
                    }

                    //Add gradient
                   // #pragma omp for private(n,i,j,z,offset,ii, index_3D, trash,P1, P2, P3, coord1, coord2, coord3, i2, j2, z2, read_file, left, middle, right, offset1, offset2, offset3)
                    for(n=0;n<n_active_total;n++){
                       if(n%num_threads==thread_id){
                            P1 = ACTIVE1 - AMR_LEVEL1[n_ord[n]];
                            P2 = ACTIVE2 - AMR_LEVEL2[n_ord[n]];
                            P3 = ACTIVE3 - AMR_LEVEL3[n_ord[n]];
                            coord1 = AMR_COORD1[n_ord[n]];
                            coord2 = AMR_COORD2[n_ord[n]];
                            coord3 = AMR_COORD3[n_ord[n]];

                            offset1=(f1-((coord1)*((int)pow(1+REF_1,P1)*bs1))%f1)%f1;
                            offset2=(f2-((coord2)*((int)pow(1+REF_2,P2)*bs2))%f2)%f2;
                            offset3=(f3-((coord3)*((int)pow(1+REF_3,P3)*bs3))%f3)%f3;

                            for(i2=(coord1)*((int)pow(1+REF_1,P1)*bs1)+offset1; i2<(coord1+1)*((int)pow(1+REF_1,P1)*bs1); i2+=f1)for(j2=(coord2)*((int)pow(1+REF_2,P2)*bs2)+offset2; j2<(coord2+1)*((int)pow(1+REF_2,P2)*bs2); j2+=f2)for(z2=(coord3)*((int)pow(1+REF_3,P3)*bs3)+offset3; z2<(coord3+1)*((int)pow(1+REF_3,P3)*bs3); z2+=f3){
                                i2/=f1;
                                j2/=f2;
                                z2/=f3;

                                if(i2>=i_min && i2<i_max && j2>=j_min && j2<j_max && z2>=z_min && z2<z_max){
                                    index_3D=(i2-i_min)*isize+(j2-j_min)*jsize+(z2-z_min);
                                    if(dir==1){
                                        offset=-(x1[index_3D]-(startx1+(i2+0.5)*_dx1));
                                        //offset = ((0.5 + Li)-0.5*pow(1+REF_1, P1))/f1;
                                    }
                                    else if(dir==2){
                                        offset=-(x2[index_3D]-(startx2+(j2+0.5)*_dx2));
                                        //offset = ((0.5 + Lj)-0.5*pow(1+REF_2, P2))/f2;
                                    }
                                    else {
                                        offset=-(x3[index_3D]-(startx3+(z2+0.5)*_dx3));
                                        //offset = ((0.5 + Lz)-0.5*pow(1+REF_3, P3))/f3;
                                    }
                                    rho[index_3D]=rho[index_3D]+offset*rho_grad[index_3D];
                                    ug[index_3D]=ug[index_3D]+offset*ug_grad[index_3D];
                                    for(ii=0; ii<4; ii++)B[ii*gridsize_3D+index_3D]=B[ii*gridsize_3D+index_3D]+offset*B_grad[ii*gridsize_3D+index_3D];
                                    for(ii=0; ii<4; ii++)uu[(ii)*gridsize_3D+index_3D]=uu[ii*gridsize_3D+index_3D]+offset*uu_grad[ii*gridsize_3D+index_3D];

                                    if(export_raytracing){
                                       Rdot[index_3D]=Rdot[index_3D]+offset*Rdot_grad[index_3D];
                                    }

                                    //Radiation variables
                                    if(RAD_M1){
                                        E_rad[index_3D]=E_rad[index_3D]+offset*E_rad_grad[index_3D];

                                        //Do velocity conversion
                                        for(ii=0; ii<4; ii++)uu_rad[ii*gridsize_3D+index_3D]=uu_rad[ii*gridsize_3D+index_3D]+offset*uu_rad_grad[ii*gridsize_3D+index_3D];
                                    }
                                    if(RESISTIVE){
                                        for(ii=0; ii<4; ii++)E[ii*gridsize_3D+index_3D]=E[ii*gridsize_3D+index_3D]+offset*E_grad[ii*gridsize_3D+index_3D];
                                    }
                                    if(TWO_T){
                                        TE[index_3D]=TE[index_3D]+offset*TE_grad[index_3D];
                                        TI[index_3D]=TI[index_3D]+offset*TI_grad[index_3D];
                                    }
                                    if(P_NUM){
                                        photon_number[index_3D]=photon_number[index_3D]+offset*photon_number_grad[index_3D];
                                    }
                                }
                                i2*=f1;
                                j2*=f2;
                                z2*=f3;
                            }
                        }
                    }
                }
            }
        }

        if(interpolate || export_raytracing==1){
            //Convert velocity
            for(n=0;n<n_active_total;n++){
                if(n%num_threads==thread_id){
                    P1 = ACTIVE1 - AMR_LEVEL1[n_ord[n]];
                    P2 = ACTIVE2 - AMR_LEVEL2[n_ord[n]];
                    P3 = ACTIVE3 - AMR_LEVEL3[n_ord[n]];
                    coord1 = AMR_COORD1[n_ord[n]];
                    coord2 = AMR_COORD2[n_ord[n]];
                    coord3 = AMR_COORD3[n_ord[n]];

                    offset1=(f1-((coord1)*((int)pow(1+REF_1,P1)*bs1))%f1)%f1;
                    offset2=(f2-((coord2)*((int)pow(1+REF_2,P2)*bs2))%f2)%f2;
                    offset3=(f3-((coord3)*((int)pow(1+REF_3,P3)*bs3))%f3)%f3;

                    for(i2=(coord1)*((int)pow(1+REF_1,P1)*bs1)+offset1; i2<(coord1+1)*((int)pow(1+REF_1,P1)*bs1); i2+=f1)for(j2=(coord2)*((int)pow(1+REF_2,P2)*bs2)+offset2; j2<(coord2+1)*((int)pow(1+REF_2,P2)*bs2); j2+=f2)for(z2=(coord3)*((int)pow(1+REF_3,P3)*bs3)+offset3; z2<(coord3+1)*((int)pow(1+REF_3,P3)*bs3); z2+=f3){
                        i2/=f1;
                        j2/=f2;
                        z2/=f3;
                        if(i2>=i_min && i2<i_max && j2>=j_min && j2<j_max && z2>=z_min && z2<z_max){
                            index_3D=(i2-i_min)*isize+(j2-j_min)*jsize+(z2-z_min);

                            //Do velocity conversion
                            if(axisym){
                                index_2D=(i2-i_min)*(j_max-j_min)+(j2-j_min);
                                gridsize_2D=(j_max-j_min)*(i_max-i_min);
                            }
                            else{
                                index_2D=index_3D;
                                gridsize_2D=gridsize_3D;
                            }

                            alpha= 1. / sqrt(-gcon[index_2D]);
                            for(ii=1;ii<4;ii++) beta[ii]=gcon[ii*gridsize_2D+index_2D] * alpha * alpha;
                            for(ii=0; ii<4; ii++)vkerr[ii]=uu[ii*gridsize_3D+index_3D];
                            gamma=gcov[(1*4+1)*gridsize_2D+index_2D]*vkerr[1]*vkerr[1] + gcov[(2*4+2)*gridsize_2D+index_2D]*vkerr[2]*vkerr[2]+ gcov[(3*4+3)*gridsize_2D+index_2D]*vkerr[3]*vkerr[3]
                                   + 2.*(gcov[(1*4+2)*gridsize_2D+index_2D]*vkerr[1]*vkerr[2] + gcov[(1*4+3)*gridsize_2D+index_2D]*vkerr[1]*vkerr[3] + gcov[(2*4+3)*gridsize_2D+index_2D]*vkerr[2]*vkerr[3]);
                            gamma=sqrt(1.+gamma);
                            uu[index_3D]=gamma/alpha;
                            for(ii=1;ii<4;ii++){
                                uu[(ii)*gridsize_3D+index_3D]= vkerr[ii] - gamma*beta[ii]/alpha ;
                            }

                            //Radiation variables
                            if(RAD_M1){
                                //Do velocity conversion
                                for(ii=0; ii<4; ii++)vkerr[ii]=uu_rad[ii*gridsize_3D+index_3D];
                                gamma=gcov[(1*4+1)*gridsize_2D+index_2D]*vkerr[1]*vkerr[1] + gcov[(2*4+2)*gridsize_2D+index_2D]*vkerr[2]*vkerr[2]+ gcov[(3*4+3)*gridsize_2D+index_2D]*vkerr[3]*vkerr[3]
                                   + 2.*(gcov[(1*4+2)*gridsize_2D+index_2D]*vkerr[1]*vkerr[2] + gcov[(1*4+3)*gridsize_2D+index_2D]*vkerr[1]*vkerr[3] + gcov[(2*4+3)*gridsize_2D+index_2D]*vkerr[2]*vkerr[3]);
                                gamma=sqrt(1.+gamma);
                                uu_rad[index_3D]= gamma/alpha;
                                for(ii=1;ii<4;ii++){
                                    uu_rad[(ii)*gridsize_3D+index_3D]= vkerr[ii] - gamma*beta[ii]/alpha ;
                                }
                            }
                        }
                        i2*=f1;
                        j2*=f2;
                        z2*=f3;
                    }
                }
            }
        }
    }

    //Free memory
    if(interpolate){
        free(rho_grad);
        free(ug_grad);
        free(uu_grad);
        free(B_grad);
        if(export_raytracing) free(Rdot_grad);
        if(RAD_M1){
            free(E_rad_grad);
            free(uu_rad_grad);
        }
         if(RESISTIVE){
            free(E_grad);
        }
        if(TWO_T){
            free(TE_grad);
            free(TI_grad);
        }
        if(P_NUM){
            free(photon_number_grad);
        }
    }
}

void kernel_griddata3D_new(int nb, int bs1new, int bs2new, int bs3new, int nb1, int nb2, int nb3, int* n_ord, float* input, float* output, int ACTIVE1, int ACTIVE2, int ACTIVE3, int * AMR_LEVEL1, int * AMR_LEVEL2, int * AMR_LEVEL3, int * AMR_COORD1, int * AMR_COORD2, int * AMR_COORD3){
    int n, Li, Lj, Lz, index_3D, i, j, z, i2, j2, z2, isize, jsize;
    int P1, P2, P3, coord1, coord2, coord3;
    int REF_1=1;
    int REF_2=1;
    int REF_3=1;
    isize=nb3*bs3new*pow(1+REF_3,ACTIVE3)*bs2new*nb2*pow(1+REF_2,ACTIVE2);
    jsize=nb3*bs3new*pow(1+REF_3,ACTIVE3);

    #pragma omp parallel for private(n,P1, P2, P3, coord1, coord2, coord3,  Li, Lj, Lz, index_3D, i, j, z, i2, j2, z2)
    for (n=0; n < nb; n++){
        P1 = ACTIVE1 - AMR_LEVEL1[n_ord[n]];
        P2 = ACTIVE2 - AMR_LEVEL2[n_ord[n]];
        P3 = ACTIVE3 - AMR_LEVEL3[n_ord[n]];
        coord1 = AMR_COORD1[n_ord[n]];
        coord2 = AMR_COORD2[n_ord[n]];
        coord3 = AMR_COORD3[n_ord[n]];

        for (Li=0; Li< pow(1+REF_1, P1); Li++)for (Lj=0; Lj< pow(1+REF_2, P2); Lj++)for (Lz=0; Lz< pow(1+REF_3, P3); Lz++){
            for(i=0;i<bs1new;i++)for(j=0;j<bs2new;j++)for(z=0;z<bs3new;z++){
                index_3D=n*bs1new*bs2new*bs3new+i*bs3new*bs2new+j*bs3new+z;
                i2=coord1*((int)pow(1+REF_1,P1)*bs1new)+i*pow(1+REF_1, P1)+Li;
                j2=coord2*((int)pow(1+REF_2,P2)*bs2new)+j*pow(1+REF_2, P2)+Lj;
                z2=coord3*((int)pow(1+REF_3,P3)*bs3new)+z*pow(1+REF_3, P3)+Lz;
                output[i2*isize+j2*jsize+z2]=input[index_3D];
            }
        }
    }
}

void kernel_griddata2D_new(int nb, int bs1new, int bs2new, int bs3new, int nb1, int nb2, int nb3, int* n_ord, float* input, float* output, int ACTIVE1, int ACTIVE2, int ACTIVE3, int * AMR_LEVEL1, int * AMR_LEVEL2, int * AMR_LEVEL3, int * AMR_COORD1, int * AMR_COORD2, int * AMR_COORD3){
    int n, Li, Lj, index_2D, i, j,  i2, j2, isize;
    int P1, P2, coord1, coord2;
    int REF_1=1;
    int REF_2=1;
    isize=bs2new*nb2*pow(1+REF_2,ACTIVE2);

    #pragma omp parallel for private(n,P1, P2,  coord1, coord2, Li, Lj, index_2D, i, j, i2, j2)
    for (n=0; n < nb; n++){
        P1 = ACTIVE1 - AMR_LEVEL1[n_ord[n]];
        P2 = ACTIVE2 - AMR_LEVEL2[n_ord[n]];
        coord1 = AMR_COORD1[n_ord[n]];
        coord2 = AMR_COORD2[n_ord[n]];

        for (Li=0; Li< pow(1+REF_1, P1); Li++)for (Lj=0; Lj< pow(1+REF_2, P2); Lj++){
            for(i=0;i<bs1new;i++)for(j=0;j<bs2new;j++){
                index_2D=n*bs1new*bs2new+i*bs2new+j;
                i2=coord1*((int)pow(1+REF_1,P1)*bs1new)+i*pow(1+REF_1, P1)+Li;
                j2=coord2*((int)pow(1+REF_2,P2)*bs2new)+j*pow(1+REF_2, P2)+Lj;
                output[i2*isize+j2]=input[index_2D];
            }
        }
    }
}

void kernel_rdump_write(int flag, int RAD_M1, int RESISTIVE, int TWO_T, int P_NUM, char *dir, int dump, int n_active_total, int f1,int f2,int f3, int nb, int bs1, int bs2, int bs3, float* rho,float* ug, float* uu, float* B,float* E,float* E_rad,float* uu_rad,float* TE,float* TI,float* photon_number,float* gcov,float* gcon, int axisym){
    int n,i,j,z, k,ii,u, u_stride, umax, gridsize_3D, index_3D, n_start,n_end;
    int float_size = sizeof(float);
    int bs1new=bs1/f1;
    int bs2new=bs2/f2;
    int bs3new=bs3/f3;
    int NPR=9+5*RAD_M1+3*RESISTIVE+2*TWO_T+1*P_NUM;
    char filename1[100],filename2[100];
    float trash[14];
    FILE *fin;
    float *temp_array;

    u_stride = 200;
    umax = (n_active_total - n_active_total % u_stride) / u_stride;
    if (n_active_total % u_stride > 0) umax += 1;
    //#pragma omp parallel for private(n,i,j,z,k, ii,u,gridsize_3D, index_3D, n_start,n_end,fin, filename1,filename2,trash,temp_array)
    for(u=0;u<umax;u++){
        temp_array=(float *)malloc(1*(NPR*bs1new*bs2new*bs3new)*sizeof(float));
        sprintf(filename2, "/dumps%d/new_dump%d", dump, u);
        sprintf(filename1, dir);
        strcat(filename1,filename2);
        fin = fopen(filename1, "wb");
        n_start=u * u_stride;
        n_end=MIN((u + 1) * u_stride,n_active_total);
        for(n=n_start; n<n_end; n++){
            for(i=0;i<bs1new;i++)for(j=0;j<bs2new;j++)for(z=0;z<bs3new;z++){
                index_3D=n*bs1new*bs2new*bs3new+i*bs3new*bs2new+j*bs3new+z;
                gridsize_3D=nb*bs1new*bs2new*bs3new;
                trash[0]=rho[index_3D];
                trash[1]=ug[index_3D];
                for(ii=0;ii<4;ii++) trash[2+ii] = uu[(ii)*gridsize_3D+index_3D];
                for(ii=0;ii<3;ii++) trash[6+ii] = B[(ii+1)*gridsize_3D+index_3D];
                if(RAD_M1==1){
                    trash[9]=E_rad[index_3D];
                    for(ii=0;ii<4;ii++) trash[10+ii] = uu_rad[(ii)*gridsize_3D+index_3D];
                }
                if(RESISTIVE==1){
                    for(ii=0;ii<3;ii++) trash[9+5*RAD_M1+ii] = E[(ii+1)*gridsize_3D+index_3D];
                }
                if(TWO_T==1){
                    trash[9+5*RAD_M1+3*RESISTIVE] = TE[index_3D];
                    trash[9+5*RAD_M1+3*RESISTIVE+1] = TI[index_3D];
                }
                if(P_NUM==1){
                    trash[9+5*RAD_M1+3*RESISTIVE+2*TWO_T] = photon_number[index_3D];
                }
                for(k=0;k<NPR;k++)temp_array[0*NPR*bs1new*bs2new*bs3new+i*bs2new*bs3new*NPR+j*bs3new*NPR+z*NPR+k]=trash[k];
            }
            fwrite(&temp_array[0*NPR*bs1new*bs2new*bs3new], float_size,NPR*bs1new*bs2new*bs3new, fin);
        }
        fclose(fin);
        free(temp_array);
    }
}

//Calculate Jacobian transformation to Cartesian coordinates from spherical coordinates
void calc_cart_c(float r, float h, float ph, float* dxdr){
    dxdr[0] = 1;
    dxdr[1] = 0;
    dxdr[2] = 0;
    dxdr[3] = 0;
    dxdr[4] = 0;
    dxdr[5] = sin(h) * cos(ph);
    dxdr[6] = r * cos(h) * cos(ph);
    dxdr[7] = -r * sin(h) * sin(ph);
    dxdr[8] = 0;
    dxdr[9] = sin(h) * sin(ph);
    dxdr[10] = r * cos(h) * sin(ph);
    dxdr[11] = r * sin(h) * cos(ph);
    dxdr[12] = 0;
    dxdr[13] = cos(h);
    dxdr[14] = -r * sin(h);
    dxdr[15] = 0;
}

void kernel_mdot(int bs1, int bs2, int bs3, int nb, int a_ndim, int b_ndim, float *a, float *b, float *c){
    int n, ii, ij, k, l;
    int gridsize=nb*bs1*bs2*bs3;
    if(a_ndim == 5 && b_ndim == 5){
        for(n=0; n<gridsize; n++){
             for (ii=0; ii<4; ii++) c[n] += a[ii*gridsize+n] * b[ii*gridsize+n];
        }
    }
    else if(a_ndim == 6 && b_ndim == 5){
        for(n=0; n<gridsize; n++){
            for (ii=0; ii<4; ii++) for (ij=0; ij<4; ij++)c[ii*gridsize+n] += a[ii*4*gridsize+ij*gridsize+n] * b[ij*gridsize+n];
        }
    }
    else if(a_ndim == 5 && b_ndim == 6){
        for(n=0; n<gridsize; n++){
            for (ii=0; ii<4; ii++) for (ij=0; ij<4; ij++)c[ii*gridsize+n] += b[ii*4*gridsize+ij*gridsize+n] * a[ij*gridsize+n];
        }
    }
    else if(a_ndim == 6 && b_ndim == 6){
        for(n=0; n<gridsize; n++){
            for(ii=0; ii<4; ii++) for(ij=0; ij<4; ij++){
                c[(ii * 4 + ij)*gridsize+n]=0.0;
                for(k=0; k<4; k++) for(l=0; l<4; l++)c[(ii * 4 + ij)*gridsize+n] += a[(k * 4 + l)*gridsize+n] * b[(ii*4+k)*gridsize+n] * b[(ij*4+l)*gridsize+n];
            }
        }
    }
}

void kernel_misc_calc(int bs1, int bs2, int bs3, int nb, int axisym, float *uu, float *B, float *E, float *bu,float *eu, float *gcov, float *bsq, float *esq, int calc_bu, int calc_eu, int calc_bsq, int calc_esq){
    int n, ii, gridsize_3D, gridsize_2D;
    float uu_local[4], eu_local[4], ed_local[4], bu_local[4],bd_local[4],B_local[4],E_local[4],gcov_local[16];
    gridsize_3D=nb*bs1*bs2*bs3;
    if(axisym==1){
        gridsize_2D=nb*bs1*bs2;
    }
    else{
        gridsize_2D=gridsize_3D;
    }
    #pragma omp parallel for private(n, ii, uu_local, bu_local,bd_local,B_local,gcov_local)
    for(n=0; n<gridsize_3D; n++){
        for (ii=0; ii<4; ii++) {
            uu_local[ii]= uu[ii*gridsize_3D+n];
            B_local[ii]= B[ii*gridsize_3D+n];
            bu_local[ii]= 0.0;
        }
        for (ii=0; ii<16; ii++) gcov_local[ii]= gcov[ii*gridsize_2D+(n/bs3)*(axisym==1)+n*(axisym==0)];
        bu_calc(B_local, uu_local, bu_local, gcov_local);
        lower_c(bu_local, bd_local, gcov_local);
        if(calc_bu==1) for (ii=0; ii<4; ii++)  bu[ii*gridsize_3D+n]=bu_local[ii];
        if(calc_bsq==1) bsq[n] = bu_local[0] * bd_local[0] + bu_local[1] * bd_local[1] + bu_local[2] * bd_local[2] + bu_local[3] * bd_local[3];
        if(calc_esq==1){
            for (ii=0; ii<4; ii++) {
                E_local[ii]= E[ii*gridsize_3D+n];
                eu_local[ii]= 0.0;
            }
            eu_calc(E_local, uu_local, eu_local, gcov_local);
            if(calc_eu==1) for (ii=0; ii<4; ii++)  eu[ii*gridsize_3D+n]=eu_local[ii];
            lower_c(eu_local, ed_local, gcov_local);
            esq[n] = eu_local[0] * ed_local[0] + eu_local[1] * ed_local[1] + eu_local[2] * ed_local[2] + eu_local[3] * ed_local[3];
        }
    }
}

void eu_calc(float *E, float *uu, float *eu, float *gcov){
    int ii;
    eu[0]=0.0;
    for (ii=0; ii<4; ii++) eu[0] += (E[1] * uu[ii] * gcov[1*4+ii] + E[2] * uu[ii] * gcov[2*4+ii] + E[3] * uu[ii] * gcov[3*4+ii]);
    for (ii=1; ii<4; ii++) eu[ii] = (E[ii] + eu[0] * uu[ii]) / (uu[0]);
}

void bu_calc(float *B, float *uu, float *bu, float *gcov){
    int ii;
    bu[0]=0.0;
    for (ii=0; ii<4; ii++) bu[0] += (B[1] * uu[ii] * gcov[1*4+ii] + B[2] * uu[ii] * gcov[2*4+ii] + B[3] * uu[ii] * gcov[3*4+ii]);
    for (ii=1; ii<4; ii++) bu[ii] = (B[ii] + bu[0] * uu[ii]) / (uu[0]);
}

void lower_c(float *uu, float *ud, float *gcov){
    int i, j;

    for(i=0; i<4; i++) ud[i]=0.0;
    for(i=0; i<4; i++) for(j=0; j<4; j++) ud[i] += gcov[i*4+j] * uu[j];
}

void Tcalcuu_c(float rho, float ug, float *uu, float *bu, float *gcov, float *gcon, float gam, float *Tuu){
    float bsq, pg, w, bd[4];
    int  kapa, nu;

    lower_c(bu, bd, gcov);
    bsq=bu[0]*bd[0]+bu[1]*bd[1]+bu[2]*bd[2]+bu[3]*bd[3];
    pg = (gam - 1) * ug;
    w = rho + ug + pg;

    for(kapa=0; kapa<4; kapa++) for(nu=0; nu<4; nu++) Tuu[kapa*4+nu] = bsq * uu[kapa] * uu[nu] + 0.5 * bsq * gcon[kapa*4+nu] - bu[kapa] * bu[nu] + w * uu[kapa] * uu[nu] ;
}

void Tcalcud_c(float rho, float ug, float *uu, float *bu, float *gcov, float *gcon, float gam, float *Tud){
    float bsq, pg, w, ud[4],bd[4];
    int  kapa, nu;

    lower_c(uu, ud, gcov);
    lower_c(bu, bd, gcov);
    bsq=bu[0]*bd[0]+bu[1]*bd[1]+bu[2]*bd[2]+bu[3]*bd[3];
    pg = (gam - 1) * ug;
    w = rho + ug + pg;

    for(kapa=0; kapa<4; kapa++) for(nu=0; nu<4; nu++) Tud[kapa*4+nu] = bsq * uu[kapa] * ud[nu] + 0.5 * bsq * (kapa==nu) - bu[kapa] * bd[nu] + w * uu[kapa] * ud[nu] + pg * (kapa==nu);
}

void kernel_calc_prec_disk(int bs1, int bs2, int bs3, int nb, int axisym, int avg, float *r, float *h, float *ph, float *rho, float *ug, float *uu, float *B, float gam, float* gcov, float* gcon, float* gdet, float* dxdxp,
float *Su_disk, float *L_disk,float *Su_corona, float *L_corona,float *Su_disk_avg, float *L_disk_avg,float *Su_corona_avg, float *L_corona_avg){
    int i, j, z, n, i1, j1, mu, nu, k, l, index_2D, index_3D, gridsize_3D, gridsize_2D, threadid,nthreads;
    float dxdr[16], Tuu[16], Tuu_tmp[16], xc[4], uu_local[4], bu_local[4], B_local[4], gcov_local[16], gcon_local[16];
    float *Su_disk_thread, *Su_corona_thread, *Su_disk_avg_thread, *Su_corona_avg_thread, *L_disk_thread, *L_corona_thread, *L_disk_avg_thread, *L_corona_avg_thread;

	#pragma omp parallel
	{
		nthreads = omp_get_num_threads();
	}
    Su_disk_thread=(float *)malloc(4*bs1*nthreads*sizeof(float));
    Su_corona_thread=(float *)malloc(4*bs1*nthreads*sizeof(float));
    L_disk_thread=(float *)malloc(16*bs1*nthreads*sizeof(float));
    L_corona_thread=(float *)malloc(16*bs1*nthreads*sizeof(float));
    Su_disk_avg_thread=(float *)malloc(4*nthreads*sizeof(float));
    Su_corona_avg_thread=(float *)malloc(4*nthreads*sizeof(float));
    L_disk_avg_thread=(float *)malloc(16*nthreads*sizeof(float));
    L_corona_avg_thread=(float *)malloc(16*nthreads*sizeof(float));

    for(n=0;n<nthreads;n++){
        for (i=0; i<bs1; i++){
            for(mu=0; mu<4; mu++){
                Su_disk_thread[(n*4*bs1)+(mu)*bs1+i]=0.0;
                Su_corona_thread[(n*4*bs1)+(mu)*bs1+i]=0.0;
                for(nu=0; nu<4; nu++){
                    L_disk_thread[(n*16*bs1)+(mu*4+nu)*bs1+i]=0.0;
                    L_corona_thread[(n*16*bs1)+(mu*4+nu)*bs1+i]=0.0;
                }
            }
       }
       for(mu=0; mu<4; mu++){
            Su_disk_avg_thread[(n*4)+mu]=0.0;
            Su_corona_avg_thread[(n*4)+mu]=0.0;
            for(nu=0; nu<4; nu++){
                L_disk_avg_thread[(n*16)+(mu*4+nu)]=0.0;
                L_corona_avg_thread[(n*16)+(mu*4+nu)]=0.0;
            }
       }
    }

    for (n=0; n<nb; n++){
        #pragma omp parallel for private(threadid,dxdr,Tuu,Tuu_tmp,xc,uu_local,bu_local,B_local,gcov_local, gcon_local, i,j, z, i1, j1, mu, nu,k, l, index_2D, index_3D, gridsize_3D, gridsize_2D)
        for (i=0; i<bs1; i++) for (j=0; j<bs2; j++) for (z=0; z<bs3; z++){
            threadid = omp_get_thread_num();
            index_3D=n*bs1*bs2*bs3+i*bs3*bs2+j*bs3+z;
            gridsize_3D=nb*bs1*bs2*bs3;
            if(axisym==1){
                index_2D=n*bs1*bs2+i*bs2+j;
                gridsize_2D=nb*bs1*bs2;
            }
            else{
                index_2D=index_3D;
                gridsize_2D=gridsize_3D;
            }

            calc_cart_c(r[index_3D], h[index_3D], ph[index_3D], dxdr);
            xc[0] = 0;
            xc[1] = r[index_3D] * sin(h[index_3D]) * cos(ph[index_3D]);
            xc[2] = r[index_3D] * sin(h[index_3D]) * sin(ph[index_3D]);
            xc[3] = r[index_3D] * cos(h[index_3D]);
            for(i1=0;i1<4;i1++) uu_local[i1]=uu[i1*gridsize_3D+index_3D];
            for(i1=0;i1<4;i1++) B_local[i1]=B[i1*gridsize_3D+index_3D];
            for(i1=0;i1<16;i1++) gcov_local[i1]=gcov[i1*gridsize_2D+index_2D];
            for(i1=0;i1<16;i1++) gcon_local[i1]=gcon[i1*gridsize_2D+index_2D];
            bu_calc(B_local, uu_local, bu_local, gcov_local);

            Tcalcuu_c(rho[index_3D], ug[index_3D], uu_local, bu_local, gcov_local, gcon_local, gam, Tuu);

            for(i1=0; i1<4; i1++) for(j1=0; j1<4; j1++){
                Tuu_tmp[i1 * 4 + j1]=0.0;
                for(k=0; k<4; k++) for(l=0; l<4; l++){
                    Tuu_tmp[i1 * 4 + j1]+=Tuu[k * 4 + l] * dxdxp[(i1*4+k)*gridsize_2D + index_2D] * dxdxp[(j1*4+l)*gridsize_2D + index_2D];
                }
            }
            for(i1=0; i1<4; i1++)for(j1=0; j1<4; j1++) Tuu[i1 * 4 + j1]=Tuu_tmp[i1 * 4 + j1];

            for(i1=0; i1<4; i1++) for(j1=0; j1<4; j1++){
                Tuu_tmp[i1 * 4 + j1]=0.0;
                for(k=0; k<4; k++) for(l=0; l<4; l++){
                    Tuu_tmp[i1 * 4 + j1]+=Tuu[k * 4 + l] * dxdr[i1*4+k] * dxdr[j1*4+l];
                }
            }
            for(i1=0; i1<4; i1++)for(j1=0; j1<4; j1++) Tuu[i1 * 4 + j1]=Tuu_tmp[i1 * 4 + j1];
            for(mu=0; mu<4; mu++){
                Su_disk_thread[(threadid*4*bs1)+mu*bs1 + i] += ((Tuu[mu*4+0]) * gdet[index_2D]);
                Su_corona_thread[(threadid*4*bs1)+mu*bs1 + i] += ((Tuu[mu*4+0]) * gdet[index_2D] * (rho[index_3D] < 0.0000000025));
                Su_disk_avg_thread[threadid*4+mu] += ((Tuu[mu*4+0]) * gdet[index_2D]);
                Su_corona_avg_thread[threadid*4+mu] += ((Tuu[mu*4+0]) * gdet[index_2D] * (rho[index_3D] < 0.0000000025) * (r[index_3D] < 750));
            }

            for(mu=0; mu<4; mu++) for(nu=0; nu<4; nu++){
                L_disk_thread[(threadid*16*bs1)+(mu*4+nu)*bs1+i] += ((xc[mu] * Tuu[nu*4+0] - xc[nu] * Tuu[mu*4+0]) * gdet[index_2D]);
                L_corona_thread[(threadid*16*bs1)+(mu*4+nu)*bs1+i] += ((xc[mu] * Tuu[nu*4+0] - xc[nu] * Tuu[mu*4+0]) * gdet[index_2D] * (rho[index_3D] < 0.0000000025));
                L_disk_avg_thread[threadid*16+(mu*4+nu)] += ((xc[mu] * Tuu[nu*4+0] - xc[nu] * Tuu[mu*4+0]) * gdet[index_2D]);
                L_corona_avg_thread[threadid*16+(mu*4+nu)] += ((xc[mu] * Tuu[nu*4+0] - xc[nu] * Tuu[mu*4+0]) * gdet[index_2D] * (rho[index_3D] < 0.0000000025) * (r[index_3D] < 750));
            }
        }
    }
    for(n=0;n<nthreads;n++){
        for (i=0; i<bs1; i++){
            for(mu=0; mu<4; mu++){
                Su_disk[(mu)*bs1+i]+= Su_disk_thread[(n*4*bs1)+(mu)*bs1+i];
                Su_corona[(mu)*bs1+i]+= Su_corona_thread[(n*4*bs1)+(mu)*bs1+i];
                for(nu=0; nu<4; nu++){
                    L_disk[(mu*4+nu)*bs1+i]+= L_disk_thread[(n*16*bs1)+(mu*4+nu)*bs1+i];
                    L_corona[(mu*4+nu)*bs1+i]+= L_corona_thread[(n*16*bs1)+(mu*4+nu)*bs1+i];
                }
            }
        }
       for(mu=0; mu<4; mu++){
            Su_disk_avg[mu]+= Su_disk_avg_thread[(n*4)+mu];
            Su_corona_avg[mu]+= Su_corona_avg_thread[(n*4)+mu];
            for(nu=0; nu<4; nu++){
                L_disk_avg[(mu*4+nu)]+= L_disk_avg_thread[(n*16)+(mu*4+nu)];
                L_corona_avg[(mu*4+nu)]+= L_corona_avg_thread[(n*16)+(mu*4+nu)];
            }
       }
    }
    free(Su_disk_thread);
    free(Su_corona_thread);
    free(L_disk_thread);
    free(L_corona_thread);
    free(Su_disk_avg_thread);
    free(Su_corona_avg_thread);
    free(L_disk_avg_thread);
    free(L_corona_avg_thread);
}

float slope_lim(float y1, float y2, float y3, float x1, float x2, float x3)
{
	float Dqm, Dqp, Dqc, s, returnval;
	/* woodward, or monotonized central, slope limiter */
	if(fabs(x2-x1)>0.000001)Dqm = (1.0)*(y2 - y1)/(x2-x1);
	else Dqm=10000.;
	if(fabs(x3-x2)>0.000001) Dqp = (1.0)*(y3 - y2)/(x3-x2);
	else Dqp=10000.;
	if(Dqm==10000. || Dqp==10000.){
	    Dqc=10000.;
	    if(Dqm==10000. && Dqp==10000.) s=-1.;
	    else s=10.;
	}
	else {
	    Dqc = 1.0*(y3 - y1)/(x3-x1);
	    s = Dqm*Dqp;
	}
	if (s <= 0.) returnval=0.;
	else {
		if (fabs(Dqm) < fabs(Dqp) && fabs(Dqm) < fabs(Dqc))
			returnval=Dqm;
		else if (fabs(Dqp) < fabs(Dqc))
			returnval=Dqp;
		else
			returnval=Dqc;
	}

	return returnval;
}

void invert_4x4(double *a, double *b){
	/*
	Description: 
	 
	Input 4x4 flattened matrix a; then, analytically do the inversion.
	Output b, the flattened + inverted 4x4 matrix.
	This assumes the matrix is invertible (det(a) != 0).

	Method is 'Laplace Expansion Theorem', i.e. https://www.geometrictools.com/Documentation/LaplaceExpansionTheorem.pdf

	*/

	double s0,s1,s2,s3,s4,s5;
	double c0,c1,c2,c3,c4,c5;
	double det_a;

	// Calculate determinants of submatrices
	s0 = a[0]*a[5] - a[4]*a[1];
	s1 = a[0]*a[6] - a[4]*a[2];
	s2 = a[0]*a[7] - a[4]*a[3];
	s3 = a[1]*a[6] - a[5]*a[2];
	s4 = a[1]*a[7] - a[5]*a[3];
	s5 = a[2]*a[7] - a[6]*a[3];

	c5 = a[10]*a[15] - a[14]*a[11];
	c4 = a[9]*a[15] - a[13]*a[11];
	c3 = a[9]*a[14] - a[13]*a[10];
	c2 = a[8]*a[15] - a[12]*a[11];
	c1 = a[8]*a[14] - a[12]*a[10];
	c0 = a[8]*a[13] - a[12]*a[9];

	// Calculate determinant of matrix a from determinants of its submatrices
	// Should include error check if det_a = 0
	det_a = s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0;

	// If det(a) != 0, then a^{-1} = adj(a)/det(a)
	// Calculate b = inv(a) = adj(a)/det(a)
	b[0] = (a[5]*c5 - a[6]*c4 + a[7]*c3  ) / det_a;
	b[1] = (-a[1]*c5 + a[2]*c4 - a[3]*c3 ) / det_a;
	b[2] = (a[13]*s5 - a[14]*s4 + a[15]*s3  ) / det_a;
	b[3] = (-a[9]*s5 + a[10]*s4 - a[11]*s3 ) / det_a;
	b[4] = (-a[4]*c5 + a[6]*c2 - a[7]*c1 ) / det_a;
	b[5] = (a[0]*c5 - a[2]*c2 + a[3]*c1  ) / det_a;
	b[6] = (-a[12]*s5 + a[14]*s2 - a[15]*s1 ) / det_a;
	b[7] = (a[8]*s5 - a[10]*s2 + a[11]*s1  ) / det_a;
	b[8] = (a[4]*c4 - a[5]*c2 + a[7]*c0  ) / det_a;
	b[9] = (-a[0]*c4 + a[1]*c2 - a[3]*c0 ) / det_a;
	b[10] = (a[12]*s4 - a[13]*s2 + a[15]*s0  ) / det_a;
	b[11] = (-a[8]*s4 + a[9]*s2 - a[11]*s0 ) / det_a;
	b[12] = (-a[4]*c3 + a[5]*c1 - a[6]*c0 ) / det_a;
	b[13] = (a[0]*c3 - a[1]*c1 + a[2]*c0  ) / det_a;
	b[14] = (-a[12]*s3 + a[13]*s1 - a[14]*s0 ) / det_a;
	b[15] = (a[8]*s3 - a[9]*s1 + a[10]*s0  ) / det_a;

} // end invert_4x4()

void kernel_invert_4x4(float *A, float *B, int nb, int bs1, int bs2, int bs3){
	/*
	Description:

	Accepts array A that holds nb*bs1*bs2*bs3 4x4 matrices. Then, loops
	through each nb*bs1*bs2*bs3 index, gets 4x4 matrix at this coordinate,
	inverts it, and fills it into the array B

	*/
	int n,i,j,k;
	int index_3D, gridsize_3D;
	int i1,j1;
	// a_tmp and b_tmp are flattened 4x4 input and inverted matrices, respectively
	double a_tmp[16], b_tmp[16];

	// Loop over each of nb*bs1*bs2*bs3 cells
    for (n=0; n<nb; n++){
            gridsize_3D=nb*bs1*bs2*bs3;
            #pragma omp parallel for private(i,j,k,index_3D, i1, j1, a_tmp, b_tmp)
            for (i=0; i<bs1; i++) for (j=0; j<bs2; j++) for (k=0; k<bs3; k++){
            //  Get index_3D and gridsize_3D for global indexing in A, B arrays
            index_3D=n*bs1*bs2*bs3+i*bs3*bs2+j*bs3+k;

            // First get flattened 4x4 matrix from A at this (n,i,j,k) cell
            for (i1=0; i1<4; i1++) for (j1=0; j1<4; j1++){
                a_tmp[i1*4 + j1] = A[(i1*4+j1)*gridsize_3D + index_3D];
            }
            // Then, invert this 4x4 matrix and load it into b_tmp
            invert_4x4(&a_tmp[0],&b_tmp[0]);
            // Now take inverted 4x4 matrix, b_tmp, and put in B at this (n,i,j,k) cell
            for (i1=0; i1<4; i1++) for (j1=0; j1<4; j1++){
                B[(i1*4+j1)*gridsize_3D + index_3D] = b_tmp[i1*4 + j1];
            }
        }
	}
} // end kernel_invert_4x4()

