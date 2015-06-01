/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/
__kernel void helloworld(__global char* in, __global char* out)
{
	int num = get_global_id(0);
	out[num] = in[num] + 1;
}

 __kernel void nbody_kern(__global float4* epsilon1, __global float4* epsilon2, __global float4* k_n,__global float* O_n, __global float* t ,__global float* x, __global float* sigma, __global float4* electricfield ) 
{
	int num = get_global_id(0);
	float4 test= (float4)(2.0f,2.0f,2.0f,2.0f);
	electricfield[num] =  5*epsilon1[0];


}

 __kernel void nbody_kern(__global float4* epsilon1, __global float4* epsilon2, __global float4* k_n,__global float* O_n, __global float* t ,__global float* x, __global float* sigma, __global float4* electricfield ) 
{
	int num = get_global_id(0);
	int j;

		electricfield[num] =  num*epsilon2[0];
	


}

	char str[80];
	/*Step1: Getting platforms and choose an available one.*/
	cl_uint numPlatforms;	//the NO. of platforms
	cl_platform_id platform = NULL;	//the chosen platform
	cl_int	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != CL_SUCCESS)
	{
		cout << "Error: Getting platforms!" << endl;
		
	}

	/*For clarity, choose the first available platform. */
	if(numPlatforms > 0)
	{
		cl_platform_id* platforms = (cl_platform_id* )malloc(numPlatforms* sizeof(cl_platform_id));
		status = clGetPlatformIDs(numPlatforms, platforms, NULL);
		platform = platforms[0];
		free(platforms);
	}

	/*Step 2:Query the platform and choose the first GPU device if has one.Otherwise use the CPU as device.*/
	cl_uint				numDevices = 0;
	cl_device_id        *devices;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);	

	if (numDevices == 0)	//no GPU available.
	{
		cout << "No GPU device available." << endl;
		cout << "Choose CPU as default device." << endl;
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);	
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
	}
	else
	{
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
	}
	

	/*Step 3: Create context.*/
	cl_context context = clCreateContext(NULL,1, devices,NULL,NULL,NULL);
	
	/*Step 4: Creating command queue associate with the context.*/
	cl_command_queue commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);

	/*Step 5: Create program object */
	const char *filename = "kernel2.cl";
	string sourceStr;
	status = convertToString(filename, sourceStr);
	const char *source = sourceStr.c_str();
	size_t sourceSize[] = {strlen(source)};
	cl_program program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);
	
	/*Step 6: Build program. */
	status=clBuildProgram(program, 1,devices,NULL,NULL,NULL);

	/*Step 7: Initial input,output for the host and create memory objects for the kernel*/


	
	
	//input1[0]=(float4)(1.0f,1.0f,1.0f,1.0f);
	//size_t strlength = strlen(input);
	//cout << "input string:" << endl;
	//cout << input << endl;


	/*Initialized one time*/
	cl_mem inputBuffer1 = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, 1*sizeof(cl_float4),(void *) input1, NULL); //epsilon1
	cl_mem inputBuffer2 = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, 1*sizeof(cl_float4),(void *) input2, NULL); //epsilon2
	cl_mem inputBuffer3 = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, 1.1*pow(10.0,6.0)*sizeof(cl_float4),(void *) input3, NULL); //k_n
	cl_mem inputBuffer4 = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, 1.1*pow(10.0,6.0)*sizeof(cl_float),(void *) input4, NULL); //O_n
	
	/*Updated every run*/
	cl_mem inputBuffer5 = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, 1*sizeof(cl_float),(void *) input5, NULL); //time
	cl_mem inputBuffer6 = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, 1*sizeof(cl_float4),(void *) input6, NULL); //position
	cl_mem inputBuffer7 = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, 1.1*pow(10.0,6.0)*sizeof(cl_float),(void *) input7, NULL); //sigma

	cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY , 100000*sizeof(cl_float4), NULL, NULL);

	/*Step 8: Create kernel object */
	cl_kernel kernel = clCreateKernel(program,"nbody_kern", NULL);

	/*Step 9: Sets Kernel arguments input.*/
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&inputBuffer1);
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&inputBuffer2);
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&inputBuffer3);
	status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&inputBuffer4);
	status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&inputBuffer5);
	status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&inputBuffer6);
	status = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&inputBuffer7);

	/*Step 9: Sets Kernel arguments output.*/
	status = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&outputBuffer);
	
	/*Step 10: Running the kernel.*/
	size_t *global_work_size = (size_t *)malloc(sizeof(size_t));
	global_work_size[0]=(size_t)100000;
	status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);

	/*Step 11: Read the cout put back to host memory.*/
	status = clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0,100000*sizeof(cl_float4), output, 0, NULL, NULL);
	
	//output[5] = '\0';	//Add the terminal character to the end of output.
	//cout << "\noutput string:" << endl;
	//cout << output << endl;

	/*Step 12: Clean the resources.*/
	status = clReleaseKernel(kernel);				//Release kernel.
	status = clReleaseProgram(program);				//Release the program object.
	//status = clReleaseMemObject(inputBuffer);		//Release mem object.
	status = clReleaseMemObject(outputBuffer);
	status = clReleaseCommandQueue(commandQueue);	//Release  Command queue.
	status = clReleaseContext(context);				//Release context.
	
	printf("%f",output[80000].s[1]);
	scanf ("%s",str);
	if (output != NULL)
	{
		free(output);
		output = NULL;
	}

	if (devices != NULL)
	{
		free(devices);
		devices = NULL;
	}

	std::cout<<"Passed!\n";

	