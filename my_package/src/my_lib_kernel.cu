#include <stdbool.h>
#include <stdio.h>

#include "my_lib_kernel.h"

#define min(a,b) ((a<b)?(a):(b))
#define max(a,b) ((a>b)?(a):(b))

#define DEBUG (0)
#ifndef BLOCKDIMX
#define BLOCKDIMX (32)
#endif
#ifndef BLOCKDIMY
#define BLOCKDIMY (16)
#endif


//forward path of our layer
__global__ void SeparableConvFlowLayer_gpu_forward_kernelfunc(
		const int nElement,
		const int w, 		const int h, 		const int channel, const int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		//const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,
		const int flow_output_b_stride, const int flow_output_c_stride, const int flow_output_h_stride, const int flow_output_w_stride,

		const float* input1,    		const float* input2,    	const float* input3, 	 float* flow_output

		)
{

	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w - filter_size + 1;
	const bool withinYbounds = h_i < h - filter_size + 1;

	const int batch_i = blockIdx.z;


	//    __syncthreads();
//	const float fillvalue =0.0f;

	if( withinXbounds && withinYbounds) {
 
		float flow_y = 0.0f;
		float sum_weights = 0.0f;
		for (  int intFilterY = 0; intFilterY < filter_size; intFilterY += 1) {
			float temp2 = input2[batch_i * input2_b_stride + intFilterY * input2_c_stride + h_i * input2_h_stride + w_i ];
			flow_y += (float)(intFilterY) * temp2 ;
			sum_weights += 			temp2;
		}
		//sum_weights = fabs(sum_weights);
		flow_y = flow_y / sum_weights - ((float)(filter_size)-1.0)/2.0;
		flow_output[batch_i * flow_output_b_stride + 1 * flow_output_c_stride+ h_i* flow_output_h_stride + w_i] = 
					fabs(sum_weights) > 0.0f ?  flow_y : -2000;

		float flow_x = 0.0f;
		float sum_weights_x = 0.0f;
		for (   int intFilterX = 0; intFilterX < filter_size; intFilterX += 1) {
			float temp3 = input3[batch_i * input3_b_stride + intFilterX * input3_c_stride + h_i * input3_h_stride + w_i ];
			flow_x += (float)(intFilterX)  * temp3;
			sum_weights_x += 		 temp3;
		}
		//sum_weights_x = fabs(sum_weights_x);
		flow_x = flow_x / sum_weights_x - ((float)(filter_size)-1.0)/2.0;
		// what if the sum_weight is less than zeros.
		flow_output[batch_i * flow_output_b_stride + 0 * flow_output_c_stride + h_i* flow_output_h_stride + w_i] =
					fabs(sum_weights_x) >0.0f ? flow_x : -2000;
	}
	return ;

}

__global__ void SeparableConvFlowLayer_gpu_backward_kernelfunc(
		const int nElement, 	   const int w, 		const int h, 		const int channel, const int filter_size,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		//const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,
		const int flow_output_b_stride, const int flow_output_c_stride, const int flow_output_h_stride, const int flow_output_w_stride,

		const float* input1,        		const float* input2,		const float* input3,
		const float* gradflow_output,    		float* gradinput1,  		float* gradinput2,  		float* gradinput3
		)
		{
	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w - filter_size + 1;
	const bool withinYbounds = h_i < h - filter_size + 1;

	const int batch_i = blockIdx.z;

	if(withinXbounds && withinYbounds){
		float flow_y = 0.0f;
		float sum_weights = 0.0f;
		
		for ( int  intFilterY = 0; intFilterY < filter_size; intFilterY += 1) {
			float temp2 = input2[batch_i * input2_b_stride + intFilterY * input2_c_stride + h_i * input2_h_stride + w_i ];
			flow_y += (float)(intFilterY) * temp2 ;
			sum_weights += 			temp2;
		}
		//flow_y = flow_y / sum_weights - ((float)(filter_size)-1.0)/2.0;
		//flow_output_data[batch_i * flow_output_b_stride + 1 * flow_output_c_stride+ h_i* flow_output_h_stride + w_i] = 
		//		sum_weights >0.0f ?  flow_y : -2000;
		//float sign = sum_weights >0.0f ? 1.0f : -1.0f;
		//sum_weights = fabs(sum_weights);
		if(fabs(sum_weights) >0.0f ){
			float gradflow_y = gradflow_output[batch_i * flow_output_b_stride + 1* flow_output_c_stride + 
								h_i * flow_output_h_stride + w_i ] ;					
			float offset = flow_y / ( sum_weights * sum_weights);
			for (int  intFilterY = 0; intFilterY < filter_size; intFilterY += 1) {
				gradinput2[batch_i * input2_b_stride + intFilterY * input2_c_stride  +  h_i * input2_h_stride + w_i ] =
							gradflow_y *  ((float)(intFilterY) / sum_weights -  offset);
			}
		}
		
		
		
		float flow_x = 0.0f;
		float sum_weights_x = 0.0f;
		for ( int  intFilterX = 0; intFilterX < filter_size; intFilterX += 1) {
			float temp3 = input3[batch_i * input3_b_stride + intFilterX * input3_c_stride + h_i * input3_h_stride + w_i ];
			flow_x += (float)(intFilterX)  * temp3;
			sum_weights_x += 		 temp3;
		}
		//flow_x = flow_x / sum_weights_x - ((float)(filter_size)-1.0)/2.0;
		//flow_output_data[batch_i * flow_output_b_stride + 0 * flow_output_c_stride + h_i* flow_output_h_stride + w_i] =
		//			sum_weights_x >0 ? flow_x : -2000;
		//float sign_x = sum_weights_x >0.0f ? 1.0f : -1.0f;
		//sum_weights_x = fabs(sum_weights_x);	
		if(fabs(sum_weights_x) > 0.0f ){
			 float gradflow_x = gradflow_output[batch_i * flow_output_b_stride + 0 * flow_output_c_stride + 
									h_i * flow_output_h_stride + w_i];
			float offset  = flow_x / (sum_weights_x * sum_weights_x);
			for ( int intFilterX = 0; intFilterX < filter_size; intFilterX += 1) {
				gradinput3[batch_i * input3_b_stride + intFilterX * input3_c_stride + h_i * input3_h_stride + w_i ] +=
						gradflow_x * ((float)(intFilterX) /sum_weights_x - offset);
			}
		}
	}
	return ;

}

int SeparableConvFlowLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch,const int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		//const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,
		const int flow_output_b_stride, const int flow_output_c_stride, const int flow_output_h_stride, const int flow_output_w_stride,

		const float* input1,    		const float* input2,    	const float* input3,   float* flow_output

		)
{
	int error = -1;

	dim3 grid;
	dim3 block;


	//		blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w  - filter_size + 1 + BLOCKDIMX - 1)/ BLOCKDIMX, (h  - filter_size + 1 + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
	//extract the data of CudaTensor and use kernel to calculate.
	SeparableConvFlowLayer_gpu_forward_kernelfunc<<<grid,block,0, stream >>>(
			nElement, //to let the nummous
			w,h,channel, filter_size,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			//output_b_stride,output_c_stride,output_h_stride,output_w_stride,
			flow_output_b_stride,flow_output_c_stride,flow_output_h_stride,flow_output_w_stride,

			input1,input2,input3, flow_output
			);

	//			THCudaCheck(cudaGetLastError());
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in SeparableConvFlowLayer_gpu_forward_kernel: %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;

}

int SeparableConvFlowLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    		const int channel,  		const int batch, const int filter_size,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		//const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,
		const int flow_output_b_stride, const int flow_output_c_stride, const int flow_output_h_stride, const int flow_output_w_stride,

		const float* input1,        		const float* input2,		const float* input3,

		const float* gradflow_output,    		float* gradinput1,  		float* gradinput2,  		float* gradinput3
		)
{

	int error = -1;

	dim3 grid;
	dim3 block;


	//blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w - filter_size + 1 + BLOCKDIMX - 1)/ BLOCKDIMX, (h  - filter_size + 1+ BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);

//    cudaMemset((void*)gradinput1, 0, input1_b_stride * batch * sizeof(float));
//    cudaMemset((void*)gradinput2, 0, input2_b_stride * batch * sizeof(float));
//    cudaMemset((void*)gradinput3, 0, input3_b_stride * batch * sizeof(float));

	SeparableConvFlowLayer_gpu_backward_kernelfunc <<<grid,block,0, stream>>>(
			nElement, //to let the nummous
			w,h,channel, filter_size,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			//output_b_stride,output_c_stride,output_h_stride,output_w_stride,
			flow_output_b_stride,flow_output_c_stride,flow_output_h_stride,flow_output_w_stride,


			input1, 			input2,         input3,  			gradflow_output,
			gradinput1, 			gradinput2,     gradinput3
			);

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateGradInput %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;

}





//forward path of our layer
__global__ void SeparableConvLayer_gpu_forward_kernelfunc(
		const int nElement,
		const int w, 		const int h, 		const int channel, const int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		const float* input1,    		const float* input2,    	const float* input3, 	float* output

		)
{

	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w - filter_size + 1;
	const bool withinYbounds = h_i < h - filter_size + 1;

	const int batch_i = blockIdx.z;


	//    __syncthreads();
//	const float fillvalue =0.0f;

	if( withinXbounds && withinYbounds) {

		for ( int c_i = 0 ; c_i < channel ; c_i ++){

			float out = 0.0f;
			for (int intFilterY = 0; intFilterY < filter_size; intFilterY += 1) {
			for (int intFilterX = 0; intFilterX < filter_size; intFilterX += 1) {
				float temp1 = input1[batch_i * input1_b_stride + c_i * input1_c_stride + (h_i + intFilterY )* input1_h_stride + (w_i + intFilterX)];
				float temp2 = input2[batch_i * input2_b_stride + intFilterY * input2_c_stride + h_i * input2_h_stride + w_i ];
				float temp3 = input3[batch_i * input3_b_stride + intFilterX * input3_c_stride + h_i * input3_h_stride + w_i ];
				out += temp1* temp2 * temp3;
			}
			}
			output[batch_i * output_b_stride + c_i* output_c_stride + h_i * output_h_stride + w_i ] = out;
		}
	}
	return ;

}

__global__ void SeparableConvLayer_gpu_backward_kernelfunc(
		const int nElement, 	   const int w, 		const int h, 		const int channel, const int filter_size,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		const float* input1,        		const float* input2,		const float* input3,
		const float* gradoutput,    		float* gradinput1,  		float* gradinput2,  		float* gradinput3
		)
		{
	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w - filter_size + 1;
	const bool withinYbounds = h_i < h - filter_size + 1;

	const int batch_i = blockIdx.z;

	if(withinXbounds && withinYbounds){

		for (int c_i = 0 ; c_i < channel ; c_i ++){
				for (int   intFilterY = 0; intFilterY < filter_size; intFilterY += 1) {
				for ( int  intFilterX = 0; intFilterX < filter_size; intFilterX += 1) {
					float temp1 = input1[batch_i * input1_b_stride + c_i * input1_c_stride + (h_i + intFilterY )* input1_h_stride + (w_i + intFilterX)];
					float temp2 = input2[batch_i * input2_b_stride + intFilterY * input2_c_stride + h_i * input2_h_stride + w_i ];
					float temp3 = input3[batch_i * input3_b_stride + intFilterX * input3_c_stride + h_i * input3_h_stride + w_i ];

					float gradout = gradoutput[batch_i * output_b_stride + c_i* output_c_stride + h_i * output_h_stride + w_i ];

					atomicAdd(&gradinput1[batch_i * input1_b_stride + c_i * input1_c_stride + (h_i + intFilterY )* input1_h_stride + (w_i + intFilterX)],
						gradout * temp2 * temp3);
					atomicAdd(&gradinput2[batch_i * input2_b_stride + intFilterY * input2_c_stride  +  h_i * input2_h_stride + w_i ],
						gradout * temp1 * temp3);
					atomicAdd(&gradinput3 [batch_i * input3_b_stride + intFilterX * input3_c_stride + h_i * input3_h_stride + w_i ] ,
						gradout * temp1 * temp2);
				}
				}
		}

	}
	return ;

}

int SeparableConvLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch,const int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		const float* input1,    		const float* input2,    	const float* input3, 	float* output

		)
{
	int error = -1;

	dim3 grid;
	dim3 block;


	//		blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w  - filter_size + 1 + BLOCKDIMX - 1)/ BLOCKDIMX, (h  - filter_size + 1 + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
	//extract the data of CudaTensor and use kernel to calculate.
	SeparableConvLayer_gpu_forward_kernelfunc<<<grid,block,0, stream >>>(
			nElement, //to let the nummous
			w,h,channel, filter_size,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,

			input1,input2,input3, output
			);

	//			THCudaCheck(cudaGetLastError());
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;

}

int SeparableConvLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    		const int channel,  		const int batch, const int filter_size,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		const float* input1,        		const float* input2,		const float* input3,

		const float* gradoutput,    		float* gradinput1,  		float* gradinput2,  		float* gradinput3
		)
{

	int error = -1;

	dim3 grid;
	dim3 block;


	//blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w - filter_size + 1 + BLOCKDIMX - 1)/ BLOCKDIMX, (h  - filter_size + 1+ BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);

//    cudaMemset((void*)gradinput1, 0, input1_b_stride * batch * sizeof(float));
//    cudaMemset((void*)gradinput2, 0, input2_b_stride * batch * sizeof(float));
//    cudaMemset((void*)gradinput3, 0, input3_b_stride * batch * sizeof(float));

	SeparableConvLayer_gpu_backward_kernelfunc <<<grid,block,0, stream>>>(
			nElement, //to let the nummous
			w,h,channel, filter_size,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,


			input1, 			input2,         input3,  			gradoutput,
			gradinput1, 			gradinput2,     gradinput3
			);

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateGradInput %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;

}




//forward path of our layer
__global__ void InterpolationLayer_gpu_forward_kernelfunc(
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,

		const float* input1,
		const float* input2,
		float* output
		)
{

	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off = batch_i * input1_b_stride;

	//    __syncthreads();
	const float fillvalue =0.0f;

	if( withinXbounds && withinYbounds) {

		float fx = input2[batch_i * input2_b_stride + 0 * input2_c_stride + h_i * input2_h_stride + w_i  ];
		float fy = input2[batch_i * input2_b_stride + 1 * input2_c_stride + h_i * input2_h_stride + w_i  ];

		float x2 = (float)(w_i) + fx;
		float y2 = (float)(h_i) + fy;

		if(x2 >= 0.0f && y2 >=0.0f && x2 < (float)w && y2 < (float)h){
			int ix2_L = int(x2);
			int iy2_T = int(y2);
			int ix2_R = min(ix2_L + 1, w - 1);
			int iy2_B = min(iy2_T + 1, h - 1);

			float alpha = x2 - ix2_L;
			float beta = y2 - iy2_T;

			for(int c_i = 0 ; c_i < channel ; c_i ++){
				float TL = input1[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_L];
				float TR = input1[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_R];
				float BL = input1[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_L];
				float BR = input1[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_R];
				output[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i] =
					(1- alpha ) *(1-beta) *TL + alpha *(1- beta) * TR + (1-alpha) *beta *BL + alpha *beta * BR;
			}
		} else{
			//the warping data is out of range, we fill it with zeros
			for(int c_i = 0 ;  c_i < channel; c_i ++){
				output[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i] = fillvalue;
			}
		}
	}

	return ;

}

__global__ void InterpolationLayer_gpu_backward_kernelfunc(
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,

		const float* input1,
		const float* input2,
		const float* gradoutput,
		float* gradinput1,
		float* gradinput2
		)
{
	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off  = batch_i * input1_b_stride;

	//    __syncthreads();

	if(withinXbounds && withinYbounds){

		float fx= input2[batch_i * input2_b_stride + 0 * input2_c_stride + h_i * input2_h_stride + w_i ];
		float fy = input2[batch_i * input2_b_stride + 1* input2_c_stride + h_i * input2_h_stride + w_i];

		float x2 = float(w_i) + fx;
		float y2 = float(h_i) + fy;

		if(x2 >= 0.0f  && y2 >= 0.0f && x2 < (float)w && y2 < (float)h){
			int ix2_L = int(x2);
			int iy2_T = int(y2);

			int ix2_R  = min(ix2_L+ 1, w - 1);
			int iy2_B = min(iy2_T + 1, h - 1);

			float alpha = x2 - ix2_L;
			float beta = y2 - iy2_T;

			for (int c_i = 0 ; c_i < channel; c_i++){
				float gradoutput_value = gradoutput[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i];

				atomicAdd( & gradinput1[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_L], gradoutput_value * ( 1- alpha) * (1- beta));
				atomicAdd( & gradinput1[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_R], gradoutput_value * alpha * (1-beta));
				atomicAdd( & gradinput1[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_L], gradoutput_value * (1-alpha ) * beta);
				atomicAdd( & gradinput1[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_R], gradoutput_value * alpha * beta);

			}

			float gamma  = iy2_B - y2;

			float bot_diff = 0.0f;
			for(int c_i =0 ; c_i< channel; c_i ++ ){
				float temp = 0;
				temp += gamma * (input1[off + c_i * input1_c_stride + iy2_T * input1_h_stride +ix2_R] -
						input1[off + c_i* input1_c_stride+ iy2_T * input1_h_stride + ix2_L]);
				temp += (1 - gamma) *( input1[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_R] -
						input1[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_L]);

				float warped_diff_value = gradoutput[off+ c_i * input1_c_stride+ h_i* input1_h_stride + w_i];
				bot_diff += warped_diff_value * temp  ;


			}
			//the gradients of the x direction/ horizontal direction
			gradinput2[batch_i * input2_b_stride + 0 * input2_c_stride + h_i * input2_h_stride + w_i] = bot_diff;

			gamma = ix2_R- x2;
			bot_diff = 0.0f;
			for(int c_i = 0 ; c_i < channel;c_i ++ ){
				float temp = 0.0f;
				temp += gamma    * (input1[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_L] -
						input1[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_L]);

				temp += (1-gamma) *( input1[off + c_i * input1_c_stride+ iy2_B* input1_h_stride+ix2_R] -
						input1[off+ c_i* input1_c_stride+ iy2_T * input1_h_stride +ix2_R]);

				float warped_diff_value = gradoutput[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i];
				bot_diff += warped_diff_value * temp;


			}
			gradinput2[batch_i * input2_b_stride + 1 * input2_c_stride + h_i * input2_h_stride + w_i]= bot_diff;

		}


	}
	return ;

}


//#define __cplusplus
//#ifdef __cplusplus
//	extern "C" {
//#endif
int InterpolationLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int batch,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,

		const float* input1,
		const float* input2,
		float* output
		)
{
	int error = -1;


	dim3 grid;
	dim3 block;


	//		blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
	//extract the data of CudaTensor and use kernel to calculate.
	InterpolationLayer_gpu_forward_kernelfunc<<<grid,block,0, stream >>>(
			nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,

			input1,input2,output
			);

	//			THCudaCheck(cudaGetLastError());
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;

}

int InterpolationLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int batch,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,

		const float* input1,
		const float* input2,
		const float* gradoutput,
		float* gradinput1,
		float* gradinput2
		)
{
	int error = -1;

	dim3 grid;
	dim3 block;

	//blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
	InterpolationLayer_gpu_backward_kernelfunc <<<grid,block,0, stream>>>(
			nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,

			input1,
			input2,
			gradoutput,
			gradinput1,
			gradinput2
			);

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpu error in BilinearSampler.updateGradInput %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;

}


//forward path of our layer
__global__ void InterpolationChLayer_gpu_forward_kernelfunc(
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,

		const float* input1,
		const float* input2,
		float* output
		)
{

	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off = batch_i * input1_b_stride;

	//    __syncthreads();
	const float fillvalue =0.0f;

	if( withinXbounds && withinYbounds) {

		float fx = input2[batch_i * input2_b_stride + 0 * input2_c_stride + h_i * input2_h_stride + w_i  ];
		float fy = input2[batch_i * input2_b_stride + 1 * input2_c_stride + h_i * input2_h_stride + w_i  ];

		float x2 = (float)(w_i) + fx;
		float y2 = (float)(h_i) + fy;

		if(x2 >= 0.0f && y2 >=0.0f && x2 < (float)w && y2 < (float)h){
			int ix2_L = int(x2);
			int iy2_T = int(y2);
			int ix2_R = min(ix2_L + 1, w - 1);
			int iy2_B = min(iy2_T + 1, h - 1);

			float alpha = x2 - ix2_L;
			float beta = y2 - iy2_T;

			for(int c_i = 0 ; c_i < channel ; c_i ++){
				float TL = input1[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_L];
				float TR = input1[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_R];
				float BL = input1[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_L];
				float BR = input1[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_R];
				output[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i] =
					(1- alpha ) *(1-beta) *TL + alpha *(1- beta) * TR + (1-alpha) *beta *BL + alpha *beta * BR;
			}
		} else{
			//the warping data is out of range, we fill it with zeros
			for(int c_i = 0 ;  c_i < channel; c_i ++){
				output[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i] = fillvalue;
			}
		}
	}

	return ;

}

__global__ void InterpolationChLayer_gpu_backward_kernelfunc(
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,

		const float* input1,
		const float* input2,
		const float* gradoutput,
		float* gradinput1,
		float* gradinput2
		)
{
	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off  = batch_i * input1_b_stride;

	//    __syncthreads();

	if(withinXbounds && withinYbounds){

		float fx= input2[batch_i * input2_b_stride + 0 * input2_c_stride + h_i * input2_h_stride + w_i ];
		float fy = input2[batch_i * input2_b_stride + 1* input2_c_stride + h_i * input2_h_stride + w_i];

		float x2 = float(w_i) + fx;
		float y2 = float(h_i) + fy;

		if(x2 >= 0.0f  && y2 >= 0.0f && x2 < (float)w && y2 < (float)h){
			int ix2_L = int(x2);
			int iy2_T = int(y2);

			int ix2_R  = min(ix2_L+ 1, w - 1);
			int iy2_B = min(iy2_T + 1, h - 1);

			float alpha = x2 - ix2_L;
			float beta = y2 - iy2_T;

			for (int c_i = 0 ; c_i < channel; c_i++){
				float gradoutput_value = gradoutput[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i];

				atomicAdd( & gradinput1[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_L], gradoutput_value * ( 1- alpha) * (1- beta));
				atomicAdd( & gradinput1[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_R], gradoutput_value * alpha * (1-beta));
				atomicAdd( & gradinput1[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_L], gradoutput_value * (1-alpha ) * beta);
				atomicAdd( & gradinput1[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_R], gradoutput_value * alpha * beta);

			}

			float gamma  = iy2_B - y2;

			float bot_diff = 0.0f;
			for(int c_i =0 ; c_i< channel; c_i ++ ){
				float temp = 0;
				temp += gamma * (input1[off + c_i * input1_c_stride + iy2_T * input1_h_stride +ix2_R] -
						input1[off + c_i* input1_c_stride+ iy2_T * input1_h_stride + ix2_L]);
				temp += (1 - gamma) *( input1[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_R] -
						input1[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_L]);

				float warped_diff_value = gradoutput[off+ c_i * input1_c_stride+ h_i* input1_h_stride + w_i];
				bot_diff += warped_diff_value * temp  ;


			}
			//the gradients of the x direction/ horizontal direction
			gradinput2[batch_i * input2_b_stride + 0 * input2_c_stride + h_i * input2_h_stride + w_i] = bot_diff;

			gamma = ix2_R- x2;
			bot_diff = 0.0f;
			for(int c_i = 0 ; c_i < channel;c_i ++ ){
				float temp = 0.0f;
				temp += gamma    * (input1[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_L] -
						input1[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_L]);

				temp += (1-gamma) *( input1[off + c_i * input1_c_stride+ iy2_B* input1_h_stride+ix2_R] -
						input1[off+ c_i* input1_c_stride+ iy2_T * input1_h_stride +ix2_R]);

				float warped_diff_value = gradoutput[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i];
				bot_diff += warped_diff_value * temp;


			}
			gradinput2[batch_i * input2_b_stride + 1 * input2_c_stride + h_i * input2_h_stride + w_i]= bot_diff;

		}


	}
	return ;

}


//#define __cplusplus
//#ifdef __cplusplus
//	extern "C" {
//#endif
int InterpolationChLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int batch,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,

		const float* input1,
		const float* input2,
		float* output
		)
{
	int error = -1;


	dim3 grid;
	dim3 block;


	//		blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
	//extract the data of CudaTensor and use kernel to calculate.
	InterpolationChLayer_gpu_forward_kernelfunc<<<grid,block,0, stream >>>(
			nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,

			input1,input2,output
			);

	//			THCudaCheck(cudaGetLastError());
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;

}

int InterpolationChLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int batch,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,

		const float* input1,
		const float* input2,
		const float* gradoutput,
		float* gradinput1,
		float* gradinput2
		)
{
	int error = -1;

	dim3 grid;
	dim3 block;

	//blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
	InterpolationChLayer_gpu_backward_kernelfunc <<<grid,block,0, stream>>>(
			nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,

			input1,
			input2,
			gradoutput,
			gradinput1,
			gradinput2
			);

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpu error in BilinearSampler.updateGradInput %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;

}


//forward path of our layer
__global__ void FilterInterpolationLayer_gpu_forward_kernelfunc(
		const int nElement,
		const int w, 		const int h, 		const int channel, const int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,

		const float* input1,    		const float* input2,    	const float* input3, 	float* output

		)
{

	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off = batch_i * input1_b_stride;


	//    __syncthreads();
//	const float fillvalue =0.0f;

	if( withinXbounds && withinYbounds) {

		float fx = input2[batch_i * input2_b_stride + 0 * input2_c_stride + h_i * input2_h_stride + w_i  ];
		float fy = input2[batch_i * input2_b_stride + 1 * input2_c_stride + h_i * input2_h_stride + w_i  ];

		float x2 = (float)(w_i) + fx;
		float y2 = (float)(h_i) + fy;


		if(x2 >= 0.0f && y2 >=0.0f && x2 <= (float)(w -1) && y2 <= (float)(h-1)
            && fabs(fx) < (float)(w)/2.0f && fabs(fy) < (float)(h)/2.0f){
			int ix2_L = int(x2) + 1 - (int)(filter_size / 2);
			int iy2_T = int(y2) + 1 - (int)(filter_size / 2);
			int ix2_R = ix2_L + filter_size;
			int iy2_B = iy2_T + filter_size;

            float alpha = x2 - (int)(x2);
            float beta = y2 - (int)(y2);


			//TODO: here is a bug that if the iy2_B or ix2_R gets out of the border, than there is no enough pixels to warp the target one.
			for (int c_i = 0 ; c_i < channel ; c_i++){

                float TL = 0.0f;
                for(int filter_j = iy2_T; filter_j <= (int)(y2); filter_j ++){
                    int _filter_j = min(max(0, filter_j), h - 1);
                    for( int filter_i = ix2_L; filter_i <= (int) ( x2) ; filter_i ++ ){
                    int _filter_i = min(max(0, filter_i ), w - 1);
                    TL += input1[off + c_i *  input1_c_stride +  _filter_j * input1_h_stride + _filter_i ] *
							input3 [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i] ;
                    }
                }

                float TR = 0.0f;
                for (int filter_j = iy2_T; filter_j <= (int) (y2); filter_j ++ ){
                    int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                for (int filter_i =  (int) (x2) + 1 ; filter_i < ix2_R; filter_i ++ ){
                    int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                    TR += input1 [off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
                        input3 [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];
                }
                }

                float BL = 0.0f;
                for (int filter_j = (int) (y2) + 1; filter_j < iy2_B; filter_j ++ ){
                    int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                for (int filter_i = ix2_L; filter_i <= (int) (x2); filter_i ++ ){
                    int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                    BL += input1 [off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
                        input3 [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];
                }
                }

                float BR = 0.0f;
                for (int filter_j = (int) (y2) + 1; filter_j < iy2_B; filter_j ++ ){
                    int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                for (int filter_i = (int) (x2) + 1; filter_i < ix2_R; filter_i ++ ){
                    int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                    BR += input1 [off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
                        input3 [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];
                }
                }

                output[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i ] =
                            (1-alpha)*(1-beta)*TL +
							alpha*(1-beta)*TR +
							(1-alpha)*beta*BL +
							alpha*beta*BR;

//					for( int filter_i = ix2_L; filter_i < ix2_R ; filter_i ++ ){
//						int _filter_i = min(max(0, filter_i),w - 1);
//						output[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i ] +=
//							input1[off + c_i *  input1_c_stride +  _filter_j * input1_h_stride + _filter_i ] *
//							input3 [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i] *
////							exp( -(fabs((float) filter_j - y2) + fabs((float) filter_i - x2)) / (float)(filter_size)); // the distance weight
//							exp( -(fabs((float) filter_j - y2) + fabs((float) filter_i - x2)) ); // the distance weight
//
////							if(w_i == 141 && h_i == 316 && c_i == 0 ){
////printf("gpu: %f, %f,%f,%f\n",input1[off + c_i *  input1_c_stride +  _filter_j * input1_h_stride + _filter_i ] ,
////input3 [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i],
////exp( -(fabs((float) filter_j - y2) + fabs((float) filter_i - x2)) / (float)(filter_size)),
////output[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i ]
//// );
////}
//
//					}
//				}
			}
		} else{
			//the warping data is out of range, we fill it with zeros
			for(int c_i = 0 ;  c_i < channel; c_i ++){
				output[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i] = input1[off + c_i* input1_c_stride+ h_i * input1_h_stride + w_i];
			}
		}
	}
	return ;

}

__global__ void FilterInterpolationLayer_gpu_backward_kernelfunc(
		const int nElement, 	   const int w, 		const int h, 		const int channel, 	const int filter_size,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,

		const float* input1,        		const float* input2,		const float* input3,
		const float* gradoutput,    		float* gradinput1,  		float* gradinput2,  		float* gradinput3
		)
		{
	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off  = batch_i * input1_b_stride;

	//    __syncthreads();

	if(withinXbounds && withinYbounds){

		float fx = input2[batch_i * input2_b_stride +  0 * input2_c_stride + h_i * input2_h_stride + w_i];
		float fy = input2[batch_i * input2_b_stride +  1 * input2_c_stride + h_i * input2_h_stride + w_i];

		float x2 = float(w_i) + fx;
		float y2 = float(h_i) + fy;

		if(x2 >= 0.0f  && y2 >= 0.0f && x2 <= (float)(w - 1) && y2 <= (float)(h -1)
            && fabs(fx) < (float)(w)/2.0f && fabs(fy) < (float)(h)/2.0f){
			int ix2_L = int(x2) + 1 - (int) (filter_size/2);
			int iy2_T = int(y2) + 1 - (int) (filter_size/2);
			int ix2_R = ix2_L + filter_size;
			int iy2_B = iy2_T + filter_size;

            float alpha = x2 - (int)(x2);
            float beta = y2  - (int)(y2);
			/***
			  Step 1: calculate the gradients for input1, i.e. the input image;
			 ***/
            /***
              STEP 3: calculate the gradients for input3, i.e. the filter
             ***/
             /***
                Step 1 and Step 3 are simultaneously computed
             ***/
			for (int c_i = 0 ; c_i < channel; c_i++){

				float gradoutput_value = gradoutput[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i];

                float TL_grad = gradoutput_value * (1-alpha ) * (1-beta);
                for(int filter_j = iy2_T; filter_j <= (int) (y2) ; filter_j ++ ){
                    int _filter_j = min(max(0, filter_j), h - 1);
                    for (int filter_i = ix2_L   ; filter_i <= (int)(x2) ; filter_i ++){
                    int _filter_i = min(max(0, filter_i), w - 1);
                    atomicAdd( &gradinput1[off +c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i ],
                                TL_grad * input3[batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) *
                                                                input3_c_stride + h_i * input3_h_stride + w_i]);
                    atomicAdd( & gradinput3[batch_i * input3_b_stride + ((filter_j - iy2_T ) * filter_size + (filter_i - ix2_L)) *
                                                                        input3_c_stride + h_i * input3_h_stride + w_i],
                                TL_grad * input1[off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i]);

                    }
                }

                float TR_grad= gradoutput_value * alpha * ( 1- beta);
                for (int filter_j = iy2_T; filter_j <= (int) (y2); filter_j ++ ){
                    int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                for (int filter_i =  (int) (x2) + 1 ; filter_i < ix2_R; filter_i ++ ){
                    int _filter_i = min(max(0, filter_i),w - 1);// only used for input1

                    atomicAdd( &gradinput1[off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i ],
                                TR_grad * input3[batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) *
                                                                input3_c_stride + h_i * input3_h_stride + w_i]);
                    atomicAdd( & gradinput3[batch_i * input3_b_stride + ((filter_j - iy2_T ) * filter_size + (filter_i - ix2_L)) *
                                                                        input3_c_stride + h_i * input3_h_stride + w_i],
                                TR_grad * input1[off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i]);

                    }
                    }

                   float BL_grad = gradoutput_value * ( 1 - alpha ) * beta;
                   for (int filter_j = (int) (y2) + 1; filter_j < iy2_B; filter_j ++ ){
                        int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                        for (int filter_i = ix2_L; filter_i <= (int) (x2); filter_i ++ ){
                            int _filter_i = min(max(0, filter_i),w - 1);// only used for input1

                        atomicAdd( &gradinput1[off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i ],
                                    BL_grad * input3[batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) *
                                                                    input3_c_stride + h_i * input3_h_stride + w_i]);
                        atomicAdd( & gradinput3[batch_i * input3_b_stride + ((filter_j - iy2_T ) * filter_size + (filter_i - ix2_L)) *
                                                                            input3_c_stride + h_i * input3_h_stride + w_i],
                                    BL_grad * input1[off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i]);

                    }
                    }

                float BR_grad = gradoutput_value * alpha * beta;
                 for (int filter_j = (int) (y2) + 1; filter_j < iy2_B; filter_j ++ ){
                    int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                    for (int filter_i = (int) (x2) + 1; filter_i < ix2_R; filter_i ++ ){
                        int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                        atomicAdd( &gradinput1[off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i ],
                                    BR_grad * input3[batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) *
                                                                    input3_c_stride + h_i * input3_h_stride + w_i]);
                        atomicAdd( & gradinput3[batch_i * input3_b_stride + ((filter_j - iy2_T ) * filter_size + (filter_i - ix2_L)) *
                                                                            input3_c_stride + h_i * input3_h_stride + w_i],
                                    BR_grad * input1[off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i]);
                        }
                }
//				for ( int filter_j = iy2_T; filter_j < iy2_B ; filter_j ++ ){
//					int _filter_j = min(max(0, filter_j),  h - 1);
//					for( int filter_i = ix2_L; filter_i< ix2_R ; filter_i++){
//						int _filter_i = min(max(0,filter_i), w - 1);
//						atomicAdd( & gradinput1[off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i],
//								gradoutput_value *
//								input3 [batch_i * input3_b_stride + ((filter_j  - iy2_T) * filter_size + (filter_i - ix2_L))* input3_c_stride + h_i * input3_h_stride + w_i] *
////								exp( -(fabs((float)filter_j - y2) + fabs((float)filter_i - x2))/(float)filter_size)
//                                exp( -(fabs((float)filter_j - y2) + fabs((float)filter_i - x2)))
//
//							 );
//					}
//				}

			}

			/***
			  Step 2: calculate the gradients for input2, i.e., the optical flow,
			  STEP 2.1: for the x/horizonotal direction.
			 ***/
            float gamma  =  1.0f - beta; //iy2_B - y2;
			float bot_diff = 0.0f;
			for(int c_i =0 ; c_i< channel; c_i ++ ){
				float gradoutput_value = gradoutput[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i];

    float TL = 0.0f;
                for(int filter_j = iy2_T; filter_j <= (int)(y2); filter_j ++){
                    int _filter_j = min(max(0, filter_j), h - 1);
                    for( int filter_i = ix2_L; filter_i <= (int) ( x2) ; filter_i ++ ){
                    int _filter_i = min(max(0, filter_i ), w - 1);
                    TL += input1[off + c_i *  input1_c_stride +  _filter_j * input1_h_stride + _filter_i ] *
							input3 [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i] ;
                    }
                }

                float TR = 0.0f;
                for (int filter_j = iy2_T; filter_j <= (int) (y2); filter_j ++ ){
                    int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                for (int filter_i =  (int) (x2) + 1 ; filter_i < ix2_R; filter_i ++ ){
                    int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                    TR += input1 [off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
                        input3 [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];
                }
                }

                float BL = 0.0f;
                for (int filter_j = (int) (y2) + 1; filter_j < iy2_B; filter_j ++ ){
                    int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                for (int filter_i = ix2_L; filter_i <= (int) (x2); filter_i ++ ){
                    int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                    BL += input1 [off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
                        input3 [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];
                }
                }

                float BR = 0.0f;
                for (int filter_j = (int) (y2) + 1; filter_j < iy2_B; filter_j ++ ){
                    int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                for (int filter_i = (int) (x2) + 1; filter_i < ix2_R; filter_i ++ ){
                    int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                    BR += input1 [off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
                        input3 [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];
                }
                }

	            float temp = 0.0f;
                temp += gamma * (TR - TL);
                temp += (1-gamma) * (BR - BL);
                bot_diff += gradoutput_value * temp;
//				for( int filter_j = iy2_T; filter_j< iy2_B; filter_j++){
//					int _filter_j = min(max(0, filter_j) , h - 1);
//					for( int filter_i = ix2_L; filter_i< ix2_R; filter_i ++){
//						int _filter_i = min(max(0,filter_i), w-1);
//
//						bot_diff +=
//							gradoutput_value *
//							input1[off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
//							input3 [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L))* input3_c_stride + h_i * input3_h_stride + w_i   ] *
////							exp( - ( fabs((float) filter_j - y2 ) + fabs((float) filter_i - x2))/ (float)filter_size) *
////							((float) filter_i > x2 ? 1.0f : -1.0f) / (float)filter_size;
//                        	exp( - ( fabs((float) filter_j - y2 ) + fabs((float) filter_i - x2))) *
//							((float) filter_i > x2 ? 1.0f : -1.0f);
//					}
//				}
			}
			//the gradients of the x direction/ horizontal direction
			gradinput2[batch_i * input2_b_stride + 0 * input2_c_stride + h_i * input2_h_stride + w_i] = bot_diff;

			/***
			  STEP 2.2: for the x/horizonotal direction.
			 ***/
            gamma =  1.0f - alpha; //ix2_R -x2;
			bot_diff = 0.0f;
			for(int c_i = 0 ; c_i < channel; c_i ++ ){
				float gradoutput_value = gradoutput [ off + c_i * input1_c_stride + h_i * input1_h_stride +w_i];

                float TL = 0.0f;
                for(int filter_j = iy2_T; filter_j <= (int)(y2); filter_j ++){
                    int _filter_j = min(max(0, filter_j), h - 1);
                    for( int filter_i = ix2_L; filter_i <= (int) ( x2) ; filter_i ++ ){
                    int _filter_i = min(max(0, filter_i ), w - 1);
                    TL += input1[off + c_i *  input1_c_stride +  _filter_j * input1_h_stride + _filter_i ] *
							input3 [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i] ;
                    }
                }

                float TR = 0.0f;
                for (int filter_j = iy2_T; filter_j <= (int) (y2); filter_j ++ ){
                    int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                for (int filter_i =  (int) (x2) + 1 ; filter_i < ix2_R; filter_i ++ ){
                    int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                    TR += input1 [off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
                        input3 [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];
                }
                }

                float BL = 0.0f;
                for (int filter_j = (int) (y2) + 1; filter_j < iy2_B; filter_j ++ ){
                    int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                for (int filter_i = ix2_L; filter_i <= (int) (x2); filter_i ++ ){
                    int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                    BL += input1 [off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
                        input3 [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];
                }
                }

                float BR = 0.0f;
                for (int filter_j = (int) (y2) + 1; filter_j < iy2_B; filter_j ++ ){
                    int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                for (int filter_i = (int) (x2) + 1; filter_i < ix2_R; filter_i ++ ){
                    int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                    BR += input1 [off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
                        input3 [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];
                }
                }

                float temp = 0.0f;
                temp += gamma * (BL - TL);
                temp += (1.0f - gamma) * ( BR - TR);
                bot_diff += gradoutput_value * temp;

//				for( int filter_j = iy2_T; filter_j < iy2_B; filter_j ++ ){
//					int _filter_j = min(max(0, filter_j), h - 1);
//					for( int filter_i = ix2_L; filter_i < ix2_R; filter_i ++){
//						int _filter_i = min(max(0, filter_i), w - 1);
//
//						bot_diff +=
//							gradoutput_value *
//							input1[off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
//							input3 [batch_i * input3_b_stride +((filter_j - iy2_T) * filter_size + ( filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i ] *
////							exp( - (fabs((float) filter_j - y2) + fabs((float) filter_i - x2))/ (float)filter_size  ) *
////							((float) filter_j > y2 ? 1.0f : - 1.0f ) / (float)filter_size;
//							exp( - (fabs((float) filter_j - y2) + fabs((float) filter_i - x2))  ) *
//							((float) filter_j > y2 ? 1.0f : - 1.0f );
//					}
//				}
			}
			gradinput2[batch_i * input2_b_stride + 1 * input2_c_stride + h_i * input2_h_stride + w_i]= bot_diff;
			/***
			  STEP 3: calculate the gradients for input3, i.e. the filter
			 ***/
//			for(int c_i  = 0 ; c_i <channel ; c_i ++ ){
//				float gradoutput_value = gradoutput[ off + c_i * input1_c_stride + h_i * input1_h_stride + w_i ];
//				for( int filter_j=  iy2_T ; filter_j < iy2_B; filter_j ++ ){
//					int _filter_j = min(max(0, filter_j), h -1 );
//					for ( int filter_i  = ix2_L; filter_i < ix2_R; filter_i ++ ){
//						int _filter_i  = min(max(0, filter_i ), w - 1);
//
//						gradinput3 [  batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L  ) ) * input3_c_stride + h_i * input3_h_stride + w_i] +=
//							gradoutput_value *
//							input1[off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
////							exp( -(fabs((float) filter_j - y2 ) + fabs((float) filter_i - x2))/ (float)filter_size);
//							exp( -(fabs((float) filter_j - y2 ) + fabs((float) filter_i - x2)));
//					}
//				}
//			}
		}
	}
	return ;

}

int FilterInterpolationLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch, const  int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,

		const float* input1,    		const float* input2,    	const float* input3, 	float* output

		)
{
	int error = -1;

	dim3 grid;
	dim3 block;


	//		blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
	//extract the data of CudaTensor and use kernel to calculate.
	FilterInterpolationLayer_gpu_forward_kernelfunc<<<grid,block,0, stream >>>(
			nElement, //to let the nummous
			w,h,channel,filter_size,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,

			input1,input2,input3, output
			);
 
	//			THCudaCheck(cudaGetLastError());
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;

}

int FilterInterpolationLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    		const int channel,  		const int batch,    		const int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,

		const float* input1,        		const float* input2,		const float* input3,

		const float* gradoutput,    		float* gradinput1,  		float* gradinput2,  		float* gradinput3
		)
{

	int error = -1;

	dim3 grid;
	dim3 block;


	//blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);

//    cudaMemset((void*)gradinput1, 0, input1_b_stride * batch * sizeof(float));
//    cudaMemset((void*)gradinput2, 0, input2_b_stride * batch * sizeof(float));
//    cudaMemset((void*)gradinput3, 0, input3_b_stride * batch * sizeof(float));

	FilterInterpolationLayer_gpu_backward_kernelfunc <<<grid,block,0, stream>>>(
			nElement, //to let the nummous
			w,h,channel,filter_size,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,


			input1, 			input2,         input3,  			gradoutput,
			gradinput1, 			gradinput2,     gradinput3
			);

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateGradInput %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;

}

//forward path of our layer
__global__ void FlowProjection_gpu_forward_kernelfunc(
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,

		const float* input1,
		float* count,
		float* output
		)
{

	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off = batch_i * input1_b_stride;

	//    __syncthreads();
//	const float fillvalue =0.0f;

	if( withinXbounds && withinYbounds) {
        float fx = input1[ off + 0 * input1_c_stride + h_i * input1_h_stride + w_i ];
        float fy = input1[ off + 1 * input1_c_stride + h_i * input1_h_stride + w_i ];

        float x2 = (float) (w_i) + fx;
        float y2 = (float) (h_i) + fy;
        if(x2>=0.0f && y2 >= 0.0f &&x2 <= (float) ( w-1) && y2 <= (float) (h -1 ) ){
            int ix2_L = (int) (x2);
            int iy2_T = (int) (y2);
            int ix2_R = min(ix2_L + 1, w - 1);
            int iy2_B = min(iy2_T + 1, h - 1);

            atomicAdd(&output[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_L ] ,-fx);
            atomicAdd(&output[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_R ],-fx);
            atomicAdd(&output[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_L ] ,-fx);
            atomicAdd(&output[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_R ],-fx);

            atomicAdd(&output[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_L] , -fy);
            atomicAdd(&output[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_R]  , -fy);
            atomicAdd(&output[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_L]  , -fy);
            atomicAdd(&output[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_R]  , -fy);

            atomicAdd(& count[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_L], 1);
            atomicAdd(& count[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_R] , 1);
            atomicAdd(& count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L] , 1);
            atomicAdd(& count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_R] , 1);
        }
	}
	return ;

}

__global__ void FlowProjectionAveraging_kernelfunc(
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,

		const float* input1,
		float* count,
		float* output
		)
{

	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off = batch_i * input1_b_stride;

	//    __syncthreads();
//	const float fillvalue =0.0f;

	if( withinXbounds && withinYbounds) {
	    float temp =count[batch_i * count_b_stride + 0 + h_i * count_h_stride + w_i] ;
        if(temp > 0.0f){
            output[off + 0 * input1_c_stride + h_i * input1_h_stride + w_i ] /= temp;
            output[off + 1 * input1_c_stride + h_i * input1_h_stride + w_i ] /= temp;
        }
	}
	return ;

}


__global__ void FlowFillhole_kernelfunc(
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,

		const float* input1,
		float* count,
		float* output
		)
{

	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off = batch_i * input1_b_stride;

	//    __syncthreads();
//	const float fillvalue =0.0f;

	if( withinXbounds && withinYbounds) {
	    float temp = count[batch_i * count_b_stride + 0 + h_i * count_h_stride + w_i] ;
        if(temp <= 0.0f){
            //search along the four directions,0/90/180/270, until finding at least one
            int left_offset = w_i;            float left_temp = 0.0f;
            while(left_temp == 0.0f && left_offset - 1 >= 0){
                left_offset = left_offset - 1;
                left_temp = count[batch_i * count_b_stride + 0 + h_i * count_h_stride + left_offset] ;
            }

            int right_offset = w_i ;            float right_temp = 0.0f;
            while(right_temp ==0.0f && right_offset + 1 <= w - 1 ){
                right_offset  = right_offset + 1 ;
                right_temp =  count[batch_i * count_b_stride + 0 + h_i * count_h_stride + right_offset] ;
            }

            int up_offset = h_i ;            float up_temp = 0.0f;
            while(up_temp == 0.0f && up_offset - 1 >=0){
                up_offset = up_offset - 1;
                up_temp =  count[batch_i * count_b_stride + 0 + up_offset * count_h_stride + w_i ] ;
            }

            int down_offset = h_i;            float down_temp = 0.0f;
            while(down_temp = 0.0f && down_offset + 1 <= h - 1 ){
                down_offset = down_offset + 1;
                down_temp =  count[batch_i * count_b_stride + 0 + down_offset * count_h_stride + w_i] ;
            }

            if(left_temp + right_temp + up_temp + down_temp <=0.0f){
                //printf("Can't fill hole, find no neighbor vectors availabel\n");
                return;
            }

            left_temp = (left_temp > 0.0f)?1:0;
            right_temp = (right_temp > 0.0f)?1:0;
            up_temp = (up_temp > 0.0f)?1:0;
            down_temp = (down_temp > 0.0f)?1:0;

            output[off + 0 * input1_c_stride + h_i * input1_h_stride + w_i ] = (
                left_temp *  output[off + 0 * input1_c_stride + h_i * input1_h_stride + left_offset] +
                right_temp *  output[off + 0 * input1_c_stride + h_i * input1_h_stride + right_offset]+
                up_temp *  output[off + 0 * input1_c_stride + up_offset * input1_h_stride + w_i] +
                down_temp *  output[off + 0 * input1_c_stride + down_offset * input1_h_stride + w_i]
            )/(
                left_temp + right_temp + up_temp + down_temp
            ) ;


            output[off + 1 * input1_c_stride + h_i * input1_h_stride + w_i ] =(
                left_temp *  output[off + 1 * input1_c_stride + h_i * input1_h_stride + left_offset] +
                right_temp *  output[off + 1 * input1_c_stride + h_i * input1_h_stride + right_offset]+
                up_temp *  output[off + 1 * input1_c_stride + up_offset * input1_h_stride + w_i] +
                down_temp *  output[off + 1 * input1_c_stride + down_offset * input1_h_stride + w_i]
            )/(
                left_temp + right_temp + up_temp + down_temp
            ) ;
        }
	}
	return ;

}
__global__ void FlowProjection_gpu_backward_kernelfunc(
		const int nElement,  	const int w, 	const int h, const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,

		const float* input1,
		const float* count,
		const float* gradoutput,
		float* gradinput1
		)
{
	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off  = batch_i * input1_b_stride;

	//    __syncthreads();

	if(withinXbounds && withinYbounds){
        float fx = input1[off + 0 * input1_c_stride + h_i * input1_h_stride + w_i] ;
        float fy = input1[off + 1 * input1_c_stride + h_i * input1_h_stride + w_i] ;

        float x2 = (float) ( w_i ) + fx;
        float y2 = (float) ( h_i ) + fy;
        if( x2 >=0.0f && y2 >= 0.0f && x2 <= (float) (w -1) && y2 <= (float) (h-1)){
            int ix2_L = (int)(x2);
            int iy2_T = (int)(y2);
            int ix2_R  = min(ix2_L + 1, w-1);
            int iy2_B  = min(iy2_T + 1, h-1);

            int iu_offset = off + 0 * input1_c_stride + h_i * input1_h_stride + w_i;
            gradinput1[iu_offset] += -  gradoutput[off +  0 * input1_c_stride + iy2_T * input1_h_stride + ix2_L]/
                                        count[batch_i * count_b_stride + 0+ iy2_T * count_h_stride + ix2_L]  ;
            gradinput1[iu_offset] += -    gradoutput[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_R ]/
                                         count[batch_i * count_b_stride +0 + iy2_T * count_h_stride  + ix2_R]          ;
            gradinput1[iu_offset ] += -  gradoutput[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_L]/
                                         count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L]  ;
            gradinput1[iu_offset ]  += -  gradoutput[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_R]/
                                         count[batch_i * count_b_stride + 0+ iy2_B * count_h_stride + ix2_R]   ;

            int iv_offset = off + 1 * input1_c_stride + h_i * input1_h_stride + w_i;
            gradinput1[iv_offset] += -  gradoutput[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_L]/
                                         count[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_L]  ;
            gradinput1[iv_offset] += - gradoutput[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_R]/
                                         count[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_R]  ;
            gradinput1[iv_offset] += -  gradoutput[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_L]/
                                    count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L]     ;
            gradinput1[iv_offset] += -  gradoutput[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_R]/
                                    count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_R]   ;
        }
	}
	return ;

}



int FlowProjection_gpu_forward_kernel(
		cudaStream_t stream, 		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch, const int fillhole,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,

		const float* input1,
		float* count,
		float* output
		)
{
    int error = -1;


	dim3 grid;
	dim3 block;


	//		blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
//    printf("I am here\n");
	//extract the data of CudaTensor and use kernel to calculate.
	FlowProjection_gpu_forward_kernelfunc<<<grid,block,0, stream >>>(
			nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			count_b_stride,count_c_stride,count_h_stride,count_w_stride,

			input1,count,output
			);
    cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}
//    printf("I am there\n");

    FlowProjectionAveraging_kernelfunc<<<grid,block,0,stream>>>(
    		nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			count_b_stride,count_c_stride,count_h_stride,count_w_stride,

			input1,count,output
    );
//    printf("I am kao\n");

	//			THCudaCheck(cudaGetLastError());
    err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}
//    printf("I am dd\n");

    if(fillhole){

//        printf("use flow fill hole\n");
        FlowFillhole_kernelfunc<<<grid,block,0,stream>>>(
    		nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			count_b_stride,count_c_stride,count_h_stride,count_w_stride,

			input1,count,output
        );

    err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		return error;
	}

    }

	error = 0;
	return error;

}

int FlowProjection_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int batch,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,

		const float* input1,
		const float* count,
		const float* gradoutput,
		float * gradinput1
		)
{

	int error = -1;

	dim3 grid;
	dim3 block;

	//blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
	FlowProjection_gpu_backward_kernelfunc <<<grid,block,0, stream>>>(
			nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			count_b_stride,count_c_stride,count_h_stride,count_w_stride,

			input1,
			count,
			gradoutput,
			gradinput1
			);
//    printf("gpu I am there\n");

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpu error in BilinearSampler.updateGradInput %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}
//    printf("gpu I am here\n");

	error = 0;
	return error;


}


//forward path of our layer
__global__ void DepthFlowProjection_gpu_forward_kernelfunc(
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
        const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
        const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,

		const float* input1,	const float* input2,
		float* count,
		float* output
		)
{

	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off = batch_i * input1_b_stride;

	//    __syncthreads();
//	const float fillvalue =0.0f;

	if( withinXbounds && withinYbounds) {
        float fx = input1[ off + 0 * input1_c_stride + h_i * input1_h_stride + w_i ];
        float fy = input1[ off + 1 * input1_c_stride + h_i * input1_h_stride + w_i ];

        float x2 = (float) (w_i) + fx;
        float y2 = (float) (h_i) + fy;
        if(x2>=0.0f && y2 >= 0.0f &&x2 <= (float) ( w-1) && y2 <= (float) (h -1 ) ){
            int ix2_L = (int) (x2);
            int iy2_T = (int) (y2);
            int ix2_R = min(ix2_L + 1, w - 1);
            int iy2_B = min(iy2_T + 1, h - 1);

            float temp = input2[batch_i * input2_b_stride + 0 + h_i * input2_h_stride + w_i];

            atomicAdd(&output[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_L ] ,- temp * fx);
            atomicAdd(&output[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_R ],-temp * fx);
            atomicAdd(&output[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_L ] ,-temp * fx);
            atomicAdd(&output[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_R ],-temp * fx);

            atomicAdd(&output[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_L] , -temp * fy);
            atomicAdd(&output[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_R]  , -temp * fy);
            atomicAdd(&output[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_L]  , -temp * fy);
            atomicAdd(&output[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_R]  , -temp * fy);

            atomicAdd(& count[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_L], temp * 1);
            atomicAdd(& count[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_R] ,temp *  1);
            atomicAdd(& count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L] , temp * 1);
            atomicAdd(& count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_R] ,temp *  1);
        }
	}
	return ;

}

__global__ void DepthFlowProjectionAveraging_kernelfunc(
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
        const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
        const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,

		const float* input1,	const float* input2,
		float* count,
		float* output
		)
{

	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off = batch_i * input1_b_stride;

	//    __syncthreads();
//	const float fillvalue =0.0f;

	if( withinXbounds && withinYbounds) {
	    float temp =count[batch_i * count_b_stride + 0 + h_i * count_h_stride + w_i] ;
        if(temp > 0.0f){
            output[off + 0 * input1_c_stride + h_i * input1_h_stride + w_i ] /= temp;
            output[off + 1 * input1_c_stride + h_i * input1_h_stride + w_i ] /= temp;
        }
	}
	return ;

}


__global__ void DepthFlowFillhole_kernelfunc(
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
        const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
        const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,

		const float* input1,	const float* input2,
		float* count,
		float* output
		)
{

	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off = batch_i * input1_b_stride;

	//    __syncthreads();
//	const float fillvalue =0.0f;

	if( withinXbounds && withinYbounds) {
	    float temp = count[batch_i * count_b_stride + 0 + h_i * count_h_stride + w_i] ;
        if(temp <= 0.0f){
            //search along the four directions,0/90/180/270, until finding at least one
            int left_offset = w_i;            float left_temp = 0.0f;
            while(left_temp == 0.0f && left_offset - 1 >= 0){
                left_offset = left_offset - 1;
                left_temp = count[batch_i * count_b_stride + 0 + h_i * count_h_stride + left_offset] ;
            }

            int right_offset = w_i ;            float right_temp = 0.0f;
            while(right_temp ==0.0f && right_offset + 1 <= w - 1 ){
                right_offset  = right_offset + 1 ;
                right_temp =  count[batch_i * count_b_stride + 0 + h_i * count_h_stride + right_offset] ;
            }

            int up_offset = h_i ;            float up_temp = 0.0f;
            while(up_temp == 0.0f && up_offset - 1 >=0){
                up_offset = up_offset - 1;
                up_temp =  count[batch_i * count_b_stride + 0 + up_offset * count_h_stride + w_i ] ;
            }

            int down_offset = h_i;            float down_temp = 0.0f;
            while(down_temp = 0.0f && down_offset + 1 <= h - 1 ){
                down_offset = down_offset + 1;
                down_temp =  count[batch_i * count_b_stride + 0 + down_offset * count_h_stride + w_i] ;
            }

            if(left_temp + right_temp + up_temp + down_temp <=0.0f){
                //printf("Can't fill hole, find no neighbor vectors availabel\n");
                return;
            }

            left_temp = (left_temp > 0.0f)?1:0;
            right_temp = (right_temp > 0.0f)?1:0;
            up_temp = (up_temp > 0.0f)?1:0;
            down_temp = (down_temp > 0.0f)?1:0;

            output[off + 0 * input1_c_stride + h_i * input1_h_stride + w_i ] = (
                left_temp *  output[off + 0 * input1_c_stride + h_i * input1_h_stride + left_offset] +
                right_temp *  output[off + 0 * input1_c_stride + h_i * input1_h_stride + right_offset]+
                up_temp *  output[off + 0 * input1_c_stride + up_offset * input1_h_stride + w_i] +
                down_temp *  output[off + 0 * input1_c_stride + down_offset * input1_h_stride + w_i]
            )/(
                left_temp + right_temp + up_temp + down_temp
            ) ;


            output[off + 1 * input1_c_stride + h_i * input1_h_stride + w_i ] =(
                left_temp *  output[off + 1 * input1_c_stride + h_i * input1_h_stride + left_offset] +
                right_temp *  output[off + 1 * input1_c_stride + h_i * input1_h_stride + right_offset]+
                up_temp *  output[off + 1 * input1_c_stride + up_offset * input1_h_stride + w_i] +
                down_temp *  output[off + 1 * input1_c_stride + down_offset * input1_h_stride + w_i]
            )/(
                left_temp + right_temp + up_temp + down_temp
            ) ;
        }
	}
	return ;

}
__global__ void DepthFlowProjection_gpu_backward_kernelfunc(
		const int nElement,  	const int w, 	const int h, const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
        const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
        const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,

		const float* input1,		const float* input2,
		const float* count,        const float* output,
		const float* gradoutput,
		float* gradinput1,
		float* gradinput2
		)
{
	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off  = batch_i * input1_b_stride;

	//    __syncthreads();

	if(withinXbounds && withinYbounds){
        float fx = input1[off + 0 * input1_c_stride + h_i * input1_h_stride + w_i] ;
        float fy = input1[off + 1 * input1_c_stride + h_i * input1_h_stride + w_i] ;

        float x2 = (float) ( w_i ) + fx;
        float y2 = (float) ( h_i ) + fy;
        if( x2 >=0.0f && y2 >= 0.0f && x2 <= (float) (w -1) && y2 <= (float) (h-1)){
            int ix2_L = (int)(x2);
            int iy2_T = (int)(y2);
            int ix2_R  = min(ix2_L + 1, w-1);
            int iy2_B  = min(iy2_T + 1, h-1);

            float temp = input2[batch_i * input2_b_stride + 0 + h_i * input2_h_stride + w_i];

            int iu_offset = off + 0 * input1_c_stride + h_i * input1_h_stride + w_i;
            gradinput1[iu_offset] += -  gradoutput[off +  0 * input1_c_stride + iy2_T * input1_h_stride + ix2_L] * temp /
                                        count[batch_i * count_b_stride + 0+ iy2_T * count_h_stride + ix2_L]  ;
            gradinput1[iu_offset] += -    gradoutput[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_R ]  * temp /
                                         count[batch_i * count_b_stride +0 + iy2_T * count_h_stride  + ix2_R]  ;
            gradinput1[iu_offset ] += -  gradoutput[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_L] * temp /
                                         count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L] ;
            gradinput1[iu_offset ]  += -  gradoutput[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_R] * temp /
                                         count[batch_i * count_b_stride + 0+ iy2_B * count_h_stride + ix2_R] ;

            int iv_offset = off + 1 * input1_c_stride + h_i * input1_h_stride + w_i;
            gradinput1[iv_offset] += -  gradoutput[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_L]  * temp /
                                         count[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_L]  ;
            gradinput1[iv_offset] += - gradoutput[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_R] * temp /
                                         count[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_R]  ;
            gradinput1[iv_offset] += -  gradoutput[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_L] * temp /
                                    count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L]     ;
            gradinput1[iv_offset] += -  gradoutput[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_R] * temp /
                                    count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_R]   ;


            int weight_offset = batch_i * input2_b_stride + 0 + h_i * input2_h_stride + w_i;
            gradinput2[weight_offset] += - gradoutput[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_L] /
                                            count[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_L] *
                                            (fx - output[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_L ] );
            gradinput2[weight_offset] += -gradoutput[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_R] /
                                            count[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_R] *
                                            (fx - output[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_R ] );
            gradinput2[weight_offset] += -gradoutput[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_L] /
                                            count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L] *
                                            (fx - output[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_L ] );
            gradinput2[weight_offset] += -gradoutput[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_R] /
                                            count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_R] *
                                            (fx - output[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_R ] );

            gradinput2[weight_offset] += - gradoutput[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_L] /
                                            count[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_L] *
                                            (fy - output[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_L ] );
            gradinput2[weight_offset] += -gradoutput[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_R] /
                                            count[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_R] *
                                            (fy - output[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_R ] );
            gradinput2[weight_offset] += -gradoutput[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_L] /
                                            count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L] *
                                            (fy - output[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_L ] );
            gradinput2[weight_offset] += -gradoutput[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_R] /
                                            count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_R] *
                                            (fy - output[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_R ] );
        }
	}
	return ;

}



int DepthFlowProjection_gpu_forward_kernel(
		cudaStream_t stream, 		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch, const int fillhole,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
        const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
        const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,

		const float* input1,	const float* input2,
		float* count,
		float* output
		)
{
    int error = -1;


	dim3 grid;
	dim3 block;


	//		blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
//    printf("I am here\n");
	//extract the data of CudaTensor and use kernel to calculate.
	DepthFlowProjection_gpu_forward_kernelfunc<<<grid,block,0, stream >>>(
			nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
            input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
            count_b_stride,count_c_stride,count_h_stride,count_w_stride,

			input1,input2,count,output
			);
    cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}
//    printf("I am there\n");

    DepthFlowProjectionAveraging_kernelfunc<<<grid,block,0,stream>>>(
    		nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
            count_b_stride,count_c_stride,count_h_stride,count_w_stride,

			input1,input2, count,output
    );
//    printf("I am kao\n");

	//			THCudaCheck(cudaGetLastError());
    err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}
//    printf("I am dd\n");

    if(fillhole){

//        printf("use flow fill hole\n");
        DepthFlowFillhole_kernelfunc<<<grid,block,0,stream>>>(
    		nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
            input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
            count_b_stride,count_c_stride,count_h_stride,count_w_stride,

			input1,input2, count,output
        );

    err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		return error;
	}

    }

	error = 0;
	return error;

}

int DepthFlowProjection_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int batch,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
        const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
        const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,

		const float* input1,		const float* input2,
		const float* count,        const float* output,
		const float* gradoutput,
		float * gradinput1,
		float* gradinput2
		)
{

	int error = -1;

	dim3 grid;
	dim3 block;

	//blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
	DepthFlowProjection_gpu_backward_kernelfunc <<<grid,block,0, stream>>>(
			nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
            count_b_stride,count_c_stride,count_h_stride,count_w_stride,

			input1, input2,
			count, output,
			gradoutput,
			gradinput1, gradinput2
			);
//    printf("gpu I am there\n");

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpu error in BilinearSampler.updateGradInput %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}
//    printf("gpu I am here\n");

	error = 0;
	return error;


}
//forward path of our layer
__global__ void WeightedFlowProjection_gpu_forward_kernelfunc(
		const int nElement,
		const int w,
		const int h,
		const int channel, const float threshhold,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
        const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,
		const int weight_b_stride,const int weight_c_stride, const int weight_h_stride, const int weight_w_stride,

		const float* input1,const float* input2, const float* input3,
		float* count, float *weight,
		float* output
		)
{

	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off = batch_i * input1_b_stride;

	//    __syncthreads();
//	const float fillvalue =0.0f;
//    if (blockIdx.z == 1 && w_i == 32 && h_i == 32){
 //       printf("\nthere is a batch 1\n");
 //   }
	if( withinXbounds && withinYbounds) {
//	    if (blockIdx.z == 1 && w_i == 32 && h_i == 32){
//        printf("\nthere is a batch 1 A\n");
 //   }

        float fx = input1[ off + 0 * input1_c_stride + h_i * input1_h_stride + w_i ];
        float fy = input1[ off + 1 * input1_c_stride + h_i * input1_h_stride + w_i ];

        float x2 = (float) (w_i) + fx;
        float y2 = (float) (h_i) + fy;
        if(x2>=0.0f && y2 >= 0.0f &&x2 <= (float) ( w-1) && y2 <= (float) (h -1 ) ){


			int x3 = (int)(max(min((float) (w_i ) + 2.0f * fx, (float) (w) - 1.0f), 0.0f));//for calculating the brightness constancy between input2 and input3
			int y3 = (int)(max(min((float) (h_i ) + 2.0f * fy, (float) (h) - 1.0f), 0.0f));
			float weight_i = 0.0f;//data1[3],data2[3];
			int channel_i;
			for(channel_i = 0; channel_i < 3; channel_i ++){
				float data1 = input2[batch_i * input2_b_stride + channel_i* input2_c_stride +
											h_i * input2_h_stride + w_i * input2_w_stride];
				float data2 = input3[batch_i * input3_b_stride + channel_i *input3_c_stride +
										y3 * input3_h_stride + x3 * input3_w_stride];
				weight_i += fabs(data1 - data2)/3.0f;
	//			    if (blockIdx.z == 1 && w_i == 32 && h_i == 32){
///				    printf("\n%d,%d, %f,%f,%f\n" , x3,y3, data1,data2,weight_i);
	//			    }
			}
			weight_i += 1e-8f; //add a small constant for better verification
    //if (blockIdx.z == 1 && w_i == 32 && h_i == 32){
     //   printf("\nthere is a batch 1 B, weight i is %f, threshold is %f\n", weight_i, threshhold);
   // }
			if(weight_i <= threshhold){
 //   if (blockIdx.z == 1 && w_i == 32 && h_i == 32){
  //      printf("\nbatch 1 is processed\n");
   // }

            int ix2_L = (int) (x2);
            int iy2_T = (int) (y2);
            int ix2_R = min(ix2_L + 1, w - 1);
            int iy2_B = min(iy2_T + 1, h - 1);

            atomicAdd(&output[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_L ],-fx);
            atomicAdd(&output[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_R ],-fx);
            atomicAdd(&output[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_L ],-fx);
            atomicAdd(&output[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_R ],-fx);

            atomicAdd(&output[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_L]  , -fy);
            atomicAdd(&output[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_R]  , -fy);
            atomicAdd(&output[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_L]  , -fy);
            atomicAdd(&output[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_R]  , -fy);

            atomicAdd(& count[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_L], 1);
            atomicAdd(& count[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_R] , 1);
            atomicAdd(& count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L] , 1);
            atomicAdd(& count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_R] , 1);

			atomicAdd(& weight[batch_i * weight_b_stride + 0 + iy2_T * weight_h_stride + ix2_L] , weight_i);
            atomicAdd(& weight[batch_i * weight_b_stride + 0 + iy2_T * weight_h_stride + ix2_R] , weight_i);
            atomicAdd(& weight[batch_i * weight_b_stride + 0 + iy2_B * weight_h_stride + ix2_L] , weight_i);
            atomicAdd(& weight[batch_i * weight_b_stride + 0 + iy2_B * weight_h_stride + ix2_R] , weight_i);

			}
        }
	}
	return ;

}

__global__ void WeightedFlowProjectionAveraging_kernelfunc(
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
        const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,
		const int weight_b_stride,const int weight_c_stride, const int weight_h_stride, const int weight_w_stride,

		const float* input1,
		float* count,float *weight,
		float* output
		)
{

	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off = batch_i * input1_b_stride;

	//    __syncthreads();
//	const float fillvalue =0.0f;

	if( withinXbounds && withinYbounds) {
	    float temp =count[batch_i * count_b_stride + 0 + h_i * count_h_stride + w_i] ;
        if(temp > 0.0f){
            output[off + 0 * input1_c_stride + h_i * input1_h_stride + w_i ] /= temp;
            output[off + 1 * input1_c_stride + h_i * input1_h_stride + w_i ] /= temp;

			weight[batch_i * weight_b_stride + 0 + h_i * weight_h_stride + w_i ] /= temp;
        }
	}
	return ;

}


__global__ void WeightedFlowFillhole_kernelfunc(
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,
		const int weight_b_stride,const int weight_c_stride, const int weight_h_stride, const int weight_w_stride,

		const float* input1,
		float* count,float *weight,
		float* output
		)
{

	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off = batch_i * input1_b_stride;

	//    __syncthreads();
//	const float fillvalue =0.0f;

	if( withinXbounds && withinYbounds) {
	    float temp = count[batch_i * count_b_stride + 0 + h_i * count_h_stride + w_i] ;
        if(temp <= 0.0f){
            //search along the four directions,0/90/180/270, until finding at least one
            int left_offset = w_i;            float left_temp = 0.0f;
            while(left_temp == 0.0f && left_offset - 1 >= 0){
                left_offset = left_offset - 1;
                left_temp = count[batch_i * count_b_stride + 0 + h_i * count_h_stride + left_offset] ;
            }

            int right_offset = w_i ;            float right_temp = 0.0f;
            while(right_temp ==0.0f && right_offset + 1 <= w - 1 ){
                right_offset  = right_offset + 1 ;
                right_temp =  count[batch_i * count_b_stride + 0 + h_i * count_h_stride + right_offset] ;
            }

            int up_offset = h_i ;            float up_temp = 0.0f;
            while(up_temp == 0.0f && up_offset - 1 >=0){
                up_offset = up_offset - 1;
                up_temp =  count[batch_i * count_b_stride + 0 + up_offset * count_h_stride + w_i ] ;
            }

            int down_offset = h_i;            float down_temp = 0.0f;
            while(down_temp = 0.0f && down_offset + 1 <= h - 1 ){
                down_offset = down_offset + 1;
                down_temp =  count[batch_i * count_b_stride + 0 + down_offset * count_h_stride + w_i] ;
            }

            if(left_temp + right_temp + up_temp + down_temp <=0.0f){
                //printf("Can't fill hole, find no neighbor vectors availabel\n");
                return;
            }

            left_temp = (left_temp > 0.0f)?1:0;
            right_temp = (right_temp > 0.0f)?1:0;
            up_temp = (up_temp > 0.0f)?1:0;
            down_temp = (down_temp > 0.0f)?1:0;

            output[off + 0 * input1_c_stride + h_i * input1_h_stride + w_i ] = (
                left_temp *  output[off + 0 * input1_c_stride + h_i * input1_h_stride + left_offset] +
                right_temp *  output[off + 0 * input1_c_stride + h_i * input1_h_stride + right_offset]+
                up_temp *  output[off + 0 * input1_c_stride + up_offset * input1_h_stride + w_i] +
                down_temp *  output[off + 0 * input1_c_stride + down_offset * input1_h_stride + w_i]
            )/(
                left_temp + right_temp + up_temp + down_temp
            ) ;


            output[off + 1 * input1_c_stride + h_i * input1_h_stride + w_i ] =(
                left_temp *  output[off + 1 * input1_c_stride + h_i * input1_h_stride + left_offset] +
                right_temp *  output[off + 1 * input1_c_stride + h_i * input1_h_stride + right_offset]+
                up_temp *  output[off + 1 * input1_c_stride + up_offset * input1_h_stride + w_i] +
                down_temp *  output[off + 1 * input1_c_stride + down_offset * input1_h_stride + w_i]
            )/(
                left_temp + right_temp + up_temp + down_temp
            ) ;
        }
	}
	return ;

}
__global__ void WeightedFlowProjection_gpu_backward_kernelfunc(
		const int nElement,  	const int w, 	const int h, const int channel, const float threshhold,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
        const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,
		const int weight_b_stride,const int weight_c_stride, const int weight_h_stride, const int weight_w_stride,

		const float* input1,const float* input2, const float* input3,
		const float* count,const float *weight,
		const float* gradoutput,
		float* gradinput1
		)
{
	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int off  = batch_i * input1_b_stride;

	//    __syncthreads();

	if(withinXbounds && withinYbounds){
        float fx = input1[off + 0 * input1_c_stride + h_i * input1_h_stride + w_i] ;
        float fy = input1[off + 1 * input1_c_stride + h_i * input1_h_stride + w_i] ;

        float x2 = (float) ( w_i ) + fx;
        float y2 = (float) ( h_i ) + fy;
        if( x2 >=0.0f && y2 >= 0.0f && x2 <= (float) (w -1) && y2 <= (float) (h-1)){

			int x3 = (int)(max(min((float) (w_i ) + 2.0f * fx, (float) (w) - 1.0f), 0.0f));//for calculating the brightness constancy between input2 and input3
			int y3 = (int)(max(min((float) (h_i ) + 2.0f * fy, (float) (h) - 1.0f), 0.0f));
			float weight_i = 0.0f;//data1[3],data2[3];
			int channel_i;
			for(channel_i = 0; channel_i < 3; channel_i ++){
				float data1 = input2[batch_i * input2_b_stride + channel_i* input2_c_stride +
											h_i * input2_h_stride + w_i * input2_w_stride];
				float data2 = input3[batch_i * input3_b_stride + channel_i *input3_c_stride +
									y3 * input3_h_stride + x3 * input3_w_stride];
				weight_i += fabs(data1 - data2)/3.0f;
			}
			weight_i += 1e-8f; //add a small constant for better verification

			if(weight_i <= threshhold){


            int ix2_L = (int)(x2);
            int iy2_T = (int)(y2);
            int ix2_R  = min(ix2_L + 1, w-1);
            int iy2_B  = min(iy2_T + 1, h-1);

            int iu_offset = off + 0 * input1_c_stride + h_i * input1_h_stride + w_i;
            gradinput1[iu_offset] += -  gradoutput[off +  0 * input1_c_stride + iy2_T * input1_h_stride + ix2_L]/
                                        count[batch_i * count_b_stride + 0+ iy2_T * count_h_stride + ix2_L]  ;
            gradinput1[iu_offset] += -    gradoutput[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_R ]/
                                         count[batch_i * count_b_stride +0 + iy2_T * count_h_stride  + ix2_R]          ;
            gradinput1[iu_offset ] += -  gradoutput[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_L]/
                                         count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L]  ;
            gradinput1[iu_offset ]  += -  gradoutput[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_R]/
                                         count[batch_i * count_b_stride + 0+ iy2_B * count_h_stride + ix2_R]   ;

            int iv_offset = off + 1 * input1_c_stride + h_i * input1_h_stride + w_i;
            gradinput1[iv_offset] += -  gradoutput[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_L]/
                                         count[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_L]  ;
            gradinput1[iv_offset] += - gradoutput[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_R]/
                                         count[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_R]  ;
            gradinput1[iv_offset] += -  gradoutput[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_L]/
                                    count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L]     ;
            gradinput1[iv_offset] += -  gradoutput[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_R]/
                                    count[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_R]   ;
			}
        }
	}
	return ;

}



int WeightedFlowProjection_gpu_forward_kernel(
		cudaStream_t stream, 		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch, const int fillhole, const float threshold,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
        const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,
		const int weight_b_stride,const int weight_c_stride, const int weight_h_stride, const int weight_w_stride,

		const float* input1, const float* input2, const float* input3,
		float* count,  float *weight,
		float* output
		)
{
    int error = -1;


	dim3 grid;
	dim3 block;


	//		blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
  //  printf("I am here, grid size %d, %d, %d\n", grid.x, grid.y, grid.z);

    //printf("\ninput2 stride %d,%d,%d,%d", 						input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride);
    //printf("\ninput3 stride %d,%d,%d,%d", 									input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);
//    printf("\ncount stride %d,%d,%d,%d", 			count_b_stride,count_c_stride,count_h_stride,count_w_stride);
  //  printf("\nweight stride %d,%d,%d,%d", 						weight_b_stride,weight_c_stride,weight_h_stride,weight_w_stride);
	//extract the data of CudaTensor and use kernel to calculate.
	WeightedFlowProjection_gpu_forward_kernelfunc<<<grid,block,0, stream >>>(
			nElement, //to let the nummous
			w,h,channel, threshold,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			count_b_stride,count_c_stride,count_h_stride,count_w_stride,
			weight_b_stride,weight_c_stride,weight_h_stride,weight_w_stride,

			input1, input2, input3, count, weight, output
			);
    cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}
//    printf("I am there\n");

    WeightedFlowProjectionAveraging_kernelfunc<<<grid,block,0,stream>>>(
    		nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			count_b_stride,count_c_stride,count_h_stride,count_w_stride,
			weight_b_stride,weight_c_stride,weight_h_stride,weight_w_stride,

			input1,count, weight, output
    );
//    printf("I am kao\n");

	//			THCudaCheck(cudaGetLastError());
    err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}
//    printf("I am dd\n");

    if(fillhole){

//        printf("use flow fill hole\n");
        WeightedFlowFillhole_kernelfunc<<<grid,block,0,stream>>>(
    		nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			count_b_stride,count_c_stride,count_h_stride,count_w_stride,
			weight_b_stride,weight_c_stride,weight_h_stride,weight_w_stride,

			input1,count,weight, output
        );

    err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		return error;
	}

    }

	error = 0;
	return error;

}

int WeightedFlowProjection_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,
		const int h,
		const int channel,
		const int batch, const float threshhold,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
        const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,
		const int	weight_b_stride,const int weight_c_stride, const int weight_h_stride, const int weight_w_stride,

		const float* input1,const float* input2, const float* input3,
		const float* count, const float * weight,
		const float* gradoutput,
		float * gradinput1
		)
{

	int error = -1;

	dim3 grid;
	dim3 block;

	//blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
	WeightedFlowProjection_gpu_backward_kernelfunc <<<grid,block,0, stream>>>(
			nElement, //to let the nummous
			w,h,channel, threshhold,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			count_b_stride,count_c_stride,count_h_stride,count_w_stride,
			weight_b_stride,weight_c_stride,weight_h_stride,weight_w_stride,

			input1, input2, input3,
			count, weight,
			gradoutput,
			gradinput1
			);
//    printf("gpu I am there\n");

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpu error in BilinearSampler.updateGradInput %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}
//    printf("gpu I am here\n");

	error = 0;
	return error;


}


//forward path of our layer
__global__ void WeightLayer_gpu_forward_kernelfunc(
		const int nElement,
		const int w, 		const int h, 		const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
	//	const int flow1_grad_b_stride,const int flow1_grad_c_stride,const int flow1_grad_h_stride,const int flow1_grad_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		const float* input1, const float* input2, const float* input3,
		 //const float * flow1_grad,
		 float* output,
		float  lambda_e, float	lambda_v, float Nw

		)
{

	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int offset = batch_i * input1_b_stride;

	//    __syncthreads();
//	const float fillvalue =0.0f;

	if( withinXbounds && withinYbounds) {
	//read the opticalflow
	float fx = input3[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i];
	float fy = input3[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i];

	//get the destination position
	float x2 = (float)(w_i) + fx;
	float y2 = (float)(h_i) + fy;

	//Guarrantee that the center position is in-border.
	if(x2 >= 0.0f && y2 >=0.0f && x2 <=  (float) (w -1) && y2  <= (float) (h-1)){
		int ix2_L = (int)(x2);
		int iy2_T = (int)(y2);
		int ix2_R = min(ix2_L+1, w - 1);
		int iy2_B = min(iy2_T+1, h - 1);

		
		float alpha = x2 - (int)(x2);
		float beta =  y2 - (int)(y2);
		
		int m;
		int n;
		
		float err_sum = 0.0f;
	//	float sv_sum = 0.0f ;
		
		// Nw must be 3, so that -1,0,1 is the range
		for(m = -1; m <= 1; m ++){
			int patch1_m = min(max(0, m + h_i), h-1);
			for(n = -1; n <= 1; n ++){
				int patch1_n = min(max(0, n + w_i), w-1);
				
				int patch2_mT = min(max(0, m + iy2_T), h-1);
				int patch2_nL = min(max(0, n + ix2_L), w-1);
				int patch2_mB = min(max(0, m + iy2_B), h-1);
				int patch2_nR = min(max(0, n + ix2_R), w-1);
				for ( int c_i = 0; c_i < channel; c_i ++){											
					float taget_data = 
					(1-alpha)*(1-beta)*input2[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nL] +
						alpha*(1-beta)*input2[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nR] +
						(1-alpha)*beta*input2[offset + c_i * input2_c_stride + patch2_mB * input2_h_stride + patch2_nL] +
							alpha*beta*input2[offset + c_i * input2_c_stride + patch2_mB * input2_h_stride + patch2_nR];
					
					err_sum += fabsf(input1[offset + c_i * input1_c_stride + patch1_m * input1_h_stride + patch1_n]
								   - taget_data);																
				}
				//sv_sum += flow1_grad[batch_i * flow1_grad_b_stride + 0 + patch1_m * flow1_grad_h_stride + patch1_n];
			}
		}
		err_sum /= (channel * Nw * Nw);
		//sv_sum /= (Nw * Nw);
		output[batch_i * output_b_stride + 0 + h_i * output_h_stride + w_i * output_w_stride]
		    					    = (1-err_sum/lambda_e)*(1-err_sum/lambda_e);
			//= expf( - err_sum/lambda_e - sv_sum/lambda_v);
		
	}
	else {
		output[batch_i * output_b_stride + 0 + h_i * output_h_stride + w_i * output_w_stride]
			= 1e-4f; //this dosen't mean that
	}
	}
	return ;

}
int WeightLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		//const int flow1_grad_b_stride,const int flow1_grad_c_stride,const int flow1_grad_h_stride,const int flow1_grad_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		const float* input1,    		const float* input2,    	const float* input3,
		//const float * flow1_grad,
			float* output,
		float  lambda_e, float	lambda_v, float Nw
		)
{
		int error = -1;

	dim3 grid;
	dim3 block;


	//		blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
	//extract the data of CudaTensor and use kernel to calculate.
	WeightLayer_gpu_forward_kernelfunc<<<grid,block,0, stream >>>(
			nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			//flow1_grad_b_stride,flow1_grad_c_stride,flow1_grad_h_stride,flow1_grad_w_stride,
			output_b_stride,  output_c_stride,   output_h_stride,   output_w_stride,

			input1,input2,input3,
			 //flow1_grad,
			 output,
			lambda_e,  lambda_v,   Nw
			);

	//			THCudaCheck(cudaGetLastError());
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;

}



__global__ void WeightLayer_gpu_backward_kernelfunc(
		const int nElement, 	   const int w, 		const int h, 		const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		//const int flow1_grad_b_stride,const int flow1_grad_c_stride,const int flow1_grad_h_stride,const int flow1_grad_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		const float* input1,     	const float* input2, const float* input3,
		//const float * flow1_grad,
		const float* output,
		const float* gradoutput, float* gradinput1, float* gradinput2, float* gradinput3,
		 //float* gradflow1_grad,
		
		float  lambda_e, float	lambda_v, float Nw

		)
		{
	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int offset  = batch_i * input1_b_stride;

	//    __syncthreads();

	if(withinXbounds && withinYbounds) {
	//read the opticalflow
	float fx = input3[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i];
	float fy = input3[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i];

	//get the destination position
	float x2 = (float)(w_i) + fx;
	float y2 = (float)(h_i) + fy;

	//Guarrantee that the center position is in-border.
	if(x2 >= 0.0f && y2 >=0.0f && x2 <=  (float) (w -1) && y2  <= (float) (h-1)){
		int ix2_L = (int)(x2);
		int iy2_T = (int)(y2);
		int ix2_R = min(ix2_L+1, w - 1);
		int iy2_B = min(iy2_T+1, h - 1);
		float alpha = x2 - (int)(x2);
		float beta =  y2 - (int)(y2);
	 
		float gradoutput_data_value = gradoutput[batch_i * output_b_stride + 0 + h_i * output_h_stride + w_i * output_w_stride];

		float grad_err_sum = -  gradoutput_data_value / (lambda_e * channel * Nw *Nw)
									* 2 *sqrtf(output [batch_i * output_b_stride + 0 + h_i * output_h_stride + w_i * output_w_stride]) ;
//
		//float grad_err_sum = -  gradoutput_data_value *
		    //                output [batch_i * output_b_stride + 0 + h_i * output_h_stride + w_i * output_w_stride] /
		     //              (lambda_e * channel * Nw *Nw) ;
		//float grad_sv_sum =  - gradoutput_data_value *
		//                    output[batch_i * output_b_stride + 0 + h_i * output_h_stride + w_i * output_w_stride]  /
		//                    (lambda_v * Nw * Nw) ;
		
		int m;
		int n;
		// Nw must be 3, so that -1,0,1 is the range
		for(m = -1; m <= 1; m ++){
			int patch1_m = min(max(0, m + h_i), h-1);
			for(n = -1; n <= 1; n ++){
				int patch1_n = min(max(0, n + w_i), w-1);
											
				int patch2_mT = min(max(0, m + iy2_T), h-1);
				int patch2_nL = min(max(0, n + ix2_L), w-1);
				int patch2_mB = min(max(0, m + iy2_B), h-1);
				int patch2_nR = min(max(0, n + ix2_R), w-1);
				
				for (int c_i = 0; c_i < channel; c_i ++){		 	
					float taget_data =
					(1-alpha)*(1-beta)*input2[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nL] +
						alpha*(1-beta)*input2[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nR] +
						(1-alpha)*beta*input2[offset + c_i * input2_c_stride + patch2_mB * input2_h_stride + patch2_nL] +
							alpha*beta*input2[offset + c_i * input2_c_stride + patch2_mB * input2_h_stride + patch2_nR] ;
					
					float i_data = input1[offset + c_i * input1_c_stride + patch1_m * input1_h_stride + patch1_n] ;

					//input1 gradients
					atomicAdd(& gradinput1[offset + c_i * input1_c_stride + patch1_m * input1_h_stride + patch1_n] ,
										( i_data > taget_data) ?  grad_err_sum : - grad_err_sum);
					 
					 //input2 gradients
					atomicAdd(& gradinput2[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nL] ,
					(1-alpha)*(1-beta)*(( i_data> taget_data) ? - grad_err_sum :  grad_err_sum) );	
					 
					atomicAdd(& gradinput2[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nR] ,
					alpha*(1-beta)*( ( i_data> taget_data) ?  - grad_err_sum :  grad_err_sum));	
					 
					atomicAdd(& gradinput2[offset + c_i * input2_c_stride + patch2_mB  * input2_h_stride + patch2_nL] ,
					(1-alpha)*beta*( ( i_data > taget_data) ?   - grad_err_sum :  grad_err_sum));	
					
					atomicAdd(& gradinput2[offset + c_i * input2_c_stride + patch2_mB * input2_h_stride + patch2_nR]  ,
					alpha*beta*( ( i_data > taget_data) ?   - grad_err_sum :  grad_err_sum));									
					 
					//input3 gradients
					float gamma  = 1.0f - beta; //iy2_B - y2;
					float temp = 0.0f;
					temp += gamma * (input2[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nR]-
							input2[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nL]);
					temp += (1-gamma) *( input2[offset + c_i * input2_c_stride + patch2_mB * input2_h_stride + patch2_nR]  -
										input2[offset + c_i * input2_c_stride + patch2_mB  * input2_h_stride + patch2_nL]);
					
					temp = temp * ( ( i_data > taget_data) ? - grad_err_sum : grad_err_sum);
					atomicAdd(& gradinput3[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i] ,
						temp);
						
					gamma = 1.0f - alpha; //ix2_R -x2;
					temp = 0.0f;
					temp += gamma * ( input2[offset + c_i * input2_c_stride + patch2_mB  * input2_h_stride + patch2_nL] - 
										input2[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nL]);
					temp += gamma *(input2[offset + c_i * input2_c_stride + patch2_mB * input2_h_stride + patch2_nR]  -
									input2[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nR] );
					temp = temp * ( ( i_data > taget_data) ? - grad_err_sum : grad_err_sum);
					atomicAdd(& gradinput3[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i] ,
						temp);														
				}
				//flow1_grad's gradients
				//sv_sum += flow1_grad[batch_i * flow1_grad_b_stride + 0 + patch1_m * flow1_grad_h_stride + patch1_n];
				//atomicAdd(& gradflow1_grad[ batch_i * flow1_grad_b_stride + 0 + patch1_m * flow1_grad_h_stride + patch1_n]  ,
				//	grad_sv_sum);
			}
		}		
	}	
	}	
	return ;

}

int WeightLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    		const int channel,  		const int batch,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
//		const int flow1_grad_b_stride,const int flow1_grad_c_stride,const int flow1_grad_h_stride,const int flow1_grad_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		const float* input1, const float* input2, const float* input3,
		 //const float * flow1_grad,
		 const float* output,

		const float* gradoutput, float* gradinput1, float* gradinput2, float* gradinput3,
		//float* gradflow1_grad,
		float  lambda_e, float	lambda_v, float Nw

		)
{
	int error = -1;

	dim3 grid;
	dim3 block;


	//blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);

//    cudaMemset((void*)gradinput1, 0, input1_b_stride * batch * sizeof(float));
//    cudaMemset((void*)gradinput2, 0, input2_b_stride * batch * sizeof(float));
//    cudaMemset((void*)gradinput3, 0, input3_b_stride * batch * sizeof(float));

	WeightLayer_gpu_backward_kernelfunc <<<grid,block,0, stream>>>(
			nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
//			flow1_grad_b_stride,flow1_grad_c_stride,flow1_grad_h_stride,flow1_grad_w_stride,
			output_b_stride,  output_c_stride,   output_h_stride,   output_w_stride,

			input1, 			input2,         input3,
			//flow1_grad,
			 output,
			gradoutput,
			gradinput1, 			gradinput2,     gradinput3,
			//gradflow1_grad,
			lambda_e,  lambda_v,   Nw
			);

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateGradInput %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;	
}

//forward path of our layer
__global__ void PixelValueLayer_gpu_forward_kernelfunc(
		const int nElement,
		const int w, 		const int h, 		const int channel,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		const float* input1, const float* input3, const float * flow_weights,	float* output,
		float	sigma_d,     float tao_r , float  Prowindow

		)
{

	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int offset = batch_i * input1_b_stride;

 

	if( withinXbounds && withinYbounds) {
		//read the opticalflow
		float fx = input3[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i];
		float fy = input3[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i];
		
		//get the destination position
		float x2 = (float)(w_i) + fx/2.0f; //the intermediate position
		float y2 = (float)(h_i) + fy/2.0f;

		//Guarrantee that the center position is in-border.
		if(x2 >= 0.0f && y2 >=0.0f && x2 <=  (float) (w -1) && y2  <= (float) (h-1)){
			int ix2_L = (int)(x2);
			int iy2_T = (int)(y2); 
			
			float alpha = x2 - (int)(x2);
			float beta =  y2  - (int)(y2);
			
			int m;
			int n;
			// we interpolate 4 pixels, should we change the sigma_d ?
			for( m = -1; m <= 2; m ++){
 				for(n = -1; n <= 2; n ++){
 					int patch2_m = min(max(0, m + iy2_T), h-1);
					int patch2_n = min(max(0, n + ix2_L), w-1);
					
//					float g_d =  expf( - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2 * sigma_d * sigma_d));
                      float g_d = (1.0f - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
                            g_d = g_d * g_d;
					float f_w = flow_weights[batch_i * flow_weights_b_stride + 0 + h_i * flow_weights_h_stride + w_i]  ;
					for( int c_i = 0 ; c_i < channel; c_i ++){								
						atomicAdd(& output[offset + c_i * output_c_stride +  patch2_m * output_h_stride + patch2_n] ,
							f_w * g_d * input1[offset + c_i * input1_c_stride + h_i * input1_h_stride + w_i]);
					}
				}												
			}											
		}
	
	}
	return ;

}
int PixelValueLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		const float* input1, const float* input3, const float * flow_weights,	float* output,
		float	sigma_d,     float tao_r , float  Prowindow
		)
{
	int error = -1;

	dim3 grid;
	dim3 block;


	// blockthread = 128;
	// the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	// the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
	//extract the data of CudaTensor and use kernel to calculate.
	PixelValueLayer_gpu_forward_kernelfunc<<<grid,block,0, stream >>>(
			nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			//input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			flow_weights_b_stride,flow_weights_c_stride,flow_weights_h_stride,flow_weights_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,

			input1, input3, flow_weights, output,
			sigma_d,      tao_r ,   Prowindow
			);

	//			THCudaCheck(cudaGetLastError());
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;

}



__global__ void PixelValueLayer_gpu_backward_kernelfunc(
		const int nElement, 	   const int w, 		const int h, 		const int channel,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		const float* input1,  const float* input3, const float * flow_weights, 
		const float* gradoutput, float * gradinput1,   float * gradinput3, float* gradflow_weights,
		
		float	sigma_d,     float tao_r , float  Prowindow

		)
		{
	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	const int offset  = batch_i * input1_b_stride;
 
	//    __syncthreads();

	if(withinXbounds && withinYbounds) {

		//read the opticalflow
		float fx = input3[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i];
		float fy = input3[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i];
		
		//get the destination position
		float x2 = (float)(w_i) + fx/2.0f; //the intermediate position
		float y2 = (float)(h_i) + fy/2.0f;

		//Guarrantee that the center position is in-border.
		if(x2 >= 0.0f && y2 >=0.0f && x2 <=  (float) (w -1) && y2  <= (float) (h-1)){
			int ix2_L = (int)(x2);
			int iy2_T = (int)(y2);
   
 			float alpha = x2 - (int)(x2);
			float beta =  y2  - (int)(y2);
			
			int m;
			int n;
			// we interpolate 4 pixels, should we change the sigma_d ?
			for( m = -1; m <= 2; m ++){
 				for(n = -1; n <= 2; n ++){
 					
					int patch2_m = min(max(0, m + iy2_T), h-1);
					int patch2_n = min(max(0, n + ix2_L), w-1);
					
//					float g_d =  expf( - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
                      float g_d = (1.0f - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
                      //      g_d = g_d * g_d;
					float f_w = flow_weights[batch_i * flow_weights_b_stride + 0 + h_i * flow_weights_h_stride + w_i]  ;
					
					for(int c_i = 0 ; c_i < channel; c_i ++){	
						float gradoutput_data_value = gradoutput[offset + c_i * input1_c_stride +  patch2_m * input1_h_stride + patch2_n];
                            //input1 gradients
						atomicAdd(& gradinput1[offset + c_i * input1_c_stride + h_i * input1_h_stride + w_i],
							gradoutput_data_value * f_w * g_d*g_d);
						
						// flow_weights_data gradients
						atomicAdd(& gradflow_weights[batch_i * flow_weights_b_stride + 0+ h_i * flow_weights_h_stride + w_i],
							gradoutput_data_value * g_d * g_d * input1[offset + c_i * input1_c_stride + h_i * input1_h_stride + w_i]);
							
						//flow gradients
						atomicAdd(& gradinput3[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i] ,
								- gradoutput_data_value * f_w * input1[offset + c_i * input1_c_stride + h_i * input1_h_stride + w_i] *
								 g_d * (n - alpha) / (  sigma_d * sigma_d) * 2.0f);
						atomicAdd(& gradinput3[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i] ,
                                - gradoutput_data_value * f_w * input1[offset + c_i * input1_c_stride + h_i * input1_h_stride + w_i] *
								 g_d * (m - beta) / (  sigma_d * sigma_d) * 2.0f);
					}
				}												
			}											
		}
			
	}	
	return ;

}
int PixelValueLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    		const int channel,  		const int batch,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		const float* input1,  const float* input3, const float * flow_weights, 

		const float* gradoutput, float * gradinput1,   float * gradinput3, float* gradflow_weights,
		float	sigma_d,     float tao_r , float  Prowindow

		)
{ 
	int error = -1;

	dim3 grid;
	dim3 block;


	//blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);

//    cudaMemset((void*)gradinput1, 0, input1_b_stride * batch * sizeof(float));
//    cudaMemset((void*)gradinput2, 0, input2_b_stride * batch * sizeof(float));
//    cudaMemset((void*)gradinput3, 0, input3_b_stride * batch * sizeof(float));

	PixelValueLayer_gpu_backward_kernelfunc <<<grid,block,0, stream>>>(
			nElement, //to let the nummous
			w,h,channel,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			//input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			flow_weights_b_stride,flow_weights_c_stride,flow_weights_h_stride,flow_weights_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,


			input1, 			      input3,  		flow_weights, 
			gradoutput,
			gradinput1, 			     gradinput3,	gradflow_weights,
			sigma_d,      tao_r ,   Prowindow

			);

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateGradInput %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;	
}	


//forward path of our layer
__global__ void PixelWeightLayer_gpu_forward_kernelfunc(
		const int nElement,
		const int w, 		const int h,
		//const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		const float* input3, const float * flow_weights,	float * output,
		float	sigma_d,     float tao_r , float  Prowindow
		)
{
	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
 
	if( withinXbounds && withinYbounds) {

		//read the opticalflow
		float fx = input3[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i];
		float fy = input3[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i];
		
		//get the destination position
		float x2 = (float)(w_i) + fx/2.0f; //the intermediate position
		float y2 = (float)(h_i) + fy/2.0f;

		//Guarrantee that the center position is in-border.
		if(x2 >= 0.0f && y2 >=0.0f && x2 <=  (float) (w -1) && y2  <= (float) (h-1)){
			int ix2_L = (int)(x2);
			int iy2_T = (int)(y2);
			
			float alpha = x2 - (int)(x2);
			float beta =  y2  - (int)(y2);
			
			int m;
			int n;
			// we interpolate 4 pixels, should we change the sigma_d ?
			for( m = -1; m <= 2; m ++){ 									
				for(n = -1; n <= 2; n ++){ 					
					int patch2_m = min(max(0, m + iy2_T), h-1);
					int patch2_n = min(max(0, n + ix2_L), w-1);
					
//					float g_d = expf( - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
                      float g_d = (1.0f - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
                            g_d = g_d * g_d;
					float f_w = flow_weights[batch_i * flow_weights_b_stride + 0 + h_i * flow_weights_h_stride + w_i]  ;
 					atomicAdd(& output[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n] ,  f_w * g_d);
 				}												
			}											
		}	
	}
	return ;

}
int PixelWeightLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 			const int batch,

		//const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		const float* input3, const float * flow_weights,	float* output,
		float	sigma_d,     float tao_r , float  Prowindow
		)
{
		int error = -1;

	dim3 grid;
	dim3 block;


	// blockthread = 128;
	// the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	// the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
	//extract the data of CudaTensor and use kernel to calculate.
	PixelWeightLayer_gpu_forward_kernelfunc<<<grid,block,0, stream >>>(
			nElement, //to let the nummous
			w,h,
			//input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			//input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			flow_weights_b_stride,flow_weights_c_stride,flow_weights_h_stride,flow_weights_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,

			input3, flow_weights, output,
			sigma_d,  tao_r,   Prowindow
			);

	//			THCudaCheck(cudaGetLastError());
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;
}



__global__ void PixelWeightLayer_gpu_backward_kernelfunc(
		const int nElement, 	   const int w, 		const int h,
		//const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		const float* input3, const float * flow_weights,  const float* output,
		const float* gradoutput, float * gradinput3, float* gradflow_weights,
        float threshhold,
		float	sigma_d,     float tao_r , float  Prowindow
		)
{
	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
 
 
	if(withinXbounds && withinYbounds) {
		//read the opticalflow
		float fx = input3[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i];
		float fy = input3[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i];
		
		//get the destination position
		float x2 = (float)(w_i) + fx/2.0f; //the intermediate position
		float y2 = (float)(h_i) + fy/2.0f;

		//Guarrantee that the center position is in-border.
		if(x2 >= 0.0f && y2 >=0.0f && x2 <=  (float) (w -1) && y2  <= (float) (h-1)){
			int ix2_L = (int)(x2);
			int iy2_T = (int)(y2);
   
			float alpha = x2 - (int)(x2);
			float beta =  y2 - (int)(y2);
			
			int m;
			int n;
			// we interpolate 4 pixels, should we change the sigma_d ?
			for( m = -1; m <= 2; m ++){
 				for(n = -1; n <= 2; n ++){
 					int patch2_m = min(max(0, m + iy2_T), h-1);
					int patch2_n = min(max(0, n + ix2_L), w-1);
					
//					float g_d = expf( - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
                    float g_d = (1.0f - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
                    //        g_d = g_d * g_d;
					float f_w = flow_weights[batch_i * flow_weights_b_stride + 0+ h_i * flow_weights_h_stride + w_i]  ;
					
					float gradoutput_data_value = gradoutput[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n];
// 					if(output[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n] < 0)
//                        printf("Error g_d ==> %f \n",output[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n] );
                    if(output[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n] < threshhold)
                    {
                        //printf("pixelweigths gpu backward, under threshhold ==> %f\n",
                         //   output[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n]);
                              continue;//to  skip its gradients
					}
					//flow1_weights gradients
					atomicAdd(&gradflow_weights[batch_i * flow_weights_b_stride + 0+ h_i * flow_weights_h_stride + w_i],
						gradoutput_data_value * g_d * g_d);
						
					//flow gradients
					atomicAdd(& gradinput3[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i],
							- gradoutput_data_value * f_w * g_d * (n - alpha) / ( sigma_d * sigma_d) * 2.0f);
					atomicAdd(& gradinput3[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i] ,
							- gradoutput_data_value * f_w *  g_d *(m - beta)  / (  sigma_d * sigma_d) * 2.0f);
 				}												
			}											
		}
	}	
	return ;

}
int PixelWeightLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    			const int batch,

		//const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,
		
		const float* input3, const float * flow_weights,	const float* output,

		const float* gradoutput,  float* gradinput3, float* gradflow_weights,
        float threshhold,
		float	sigma_d,     float tao_r , float  Prowindow

		)
{
    int error = -1;

	dim3 grid;
	dim3 block;


	//blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);

//    cudaMemset((void*)gradinput1, 0, input1_b_stride * batch * sizeof(float));
//    cudaMemset((void*)gradinput2, 0, input2_b_stride * batch * sizeof(float));
//    cudaMemset((void*)gradinput3, 0, input3_b_stride * batch * sizeof(float));

	PixelWeightLayer_gpu_backward_kernelfunc <<<grid,block,0, stream>>>(
			nElement, //to let the nummous
			w,h,
			//input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			//input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			flow_weights_b_stride,flow_weights_c_stride,flow_weights_h_stride,flow_weights_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,


			input3,  flow_weights, output,
			gradoutput, gradinput3, gradflow_weights,
            threshhold,
            sigma_d,      tao_r ,   Prowindow

			);

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateGradInput %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;		
}



//forward path of our layer
__global__ void ReliableWeightLayer_gpu_forward_kernelfunc(
		const int nElement,
		const int w, 		const int h,

		//const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		//const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		const float* input3, 	float * output,
		float	sigma_d,     float tao_r , float  Prowindow
		)
{
	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	//only use one dimensioon of the grid and block
	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	//const int off = batch_i * input1_b_stride;


	//    __syncthreads();
//	const float fillvalue =0.0f;

	if( withinXbounds && withinYbounds) {
			//read the opticalflow
		float fx = input3[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i];
		float fy = input3[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i];

		//get the destination position
		float x2 = (float)(w_i) + fx/2.0f; //the intermediate position
		float y2 = (float)(h_i) + fy/2.0f;
		//G	uarrantee that the center position is in-border.
		if(x2 >= 0.0f && y2 >=0.0f && x2 <=  (float) (w -1) && y2  <= (float) (h-1)){
			int ix2_L = (int)(x2);
			int iy2_T = (int)(y2);
  			
			float alpha = x2 - (int)(x2);
			float beta =  y2  - (int)(y2);
			
			int m;
			int n;
			// we interpolate 4 pixels, should we change the sigma_d ?
			for( m = -1; m <= 2; m ++){										
				for(n = -1; n <= 2; n ++){				
					int patch2_m = min(max(0, m + iy2_T), h-1);
					int patch2_n = min(max(0, n + ix2_L), w-1);					
//					float g_d =  expf( - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
                    float g_d = (1.0f - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
                            g_d = g_d * g_d;
  					atomicAdd(&output[batch_i * output_b_stride + 0 +  patch2_m * output_h_stride + patch2_n], g_d);
 				}												
			}											
		}else{
			;
 		}
	}
	return ;

}
int ReliableWeightLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, const int batch,

		//const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		//const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		const float* input3,  	float* output,
		float	sigma_d,     float tao_r , float  Prowindow
		)
{
	int error = -1;

	dim3 grid;
	dim3 block;


	// blockthread = 128;
	// the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	// the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);
	//extract the data of CudaTensor and use kernel to calculate.
	ReliableWeightLayer_gpu_forward_kernelfunc<<<grid,block,0, stream >>>(
			nElement, //to let the nummous
			w,h,
			//input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			//input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			//flow_weights_b_stride,flow_weights_c_stride,flow_weights_h_stride,flow_weights_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,

			input3,  output,
			sigma_d,  tao_r,   Prowindow
			);

	//			THCudaCheck(cudaGetLastError());
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;
}



__global__ void ReliableWeightLayer_gpu_backward_kernelfunc(
		const int nElement, 	   const int w, 		const int h,
		//const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		//const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		const float* input3,const float* output,
		const float* gradoutput, float * gradinput3,  
        float threshhold,
		float	sigma_d,     float tao_r , float  Prowindow
		)
		{
	//blockIdx.z : batch index from 0~B-1
	//blockIdx.y : height patch index from ceil(h/16)
	//blockIdx.x : width patch index from ceil(w/32)

	//threadidx.x: width index 0~31
	//threadIdx.y: height index 0~15
	//threadIdx.z: Not used

	const int w_i = blockIdx.x * blockDim.x + threadIdx.x;
	const int h_i = blockIdx.y * blockDim.y + threadIdx.y;
	const bool withinXbounds = w_i < w;
	const bool withinYbounds = h_i < h;

	const int batch_i = blockIdx.z;
	//const int off  = batch_i * input1_b_stride;

	//    __syncthreads();

	if(withinXbounds && withinYbounds) 
	{
				//read the opticalflow
		float fx = input3[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i];
		float fy = input3[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i];
		
		//get the destination position
		float x2 = (float)(w_i) + fx/2.0f; //the intermediate position
		float y2 = (float)(h_i) + fy/2.0f;
		//Guarrantee that the center position is in-border.
		if(x2 >= 0.0f && y2 >=0.0f && x2 <=  (float) (w -1) && y2  <= (float) (h-1)){
			int ix2_L = (int)(x2);
			int iy2_T = (int)(y2);

			float alpha = x2 - (int)(x2);
			float beta =  y2  - (int)(y2);
			
			int m;
			int n;
			// we interpolate 4 pixels, should we change the sigma_d ?
			for( m = -1; m <= 2; m ++){
				for(n = -1; n <= 2; n ++){
					
					int patch2_m = min(max(0, m + iy2_T), h-1);
					int patch2_n = min(max(0, n + ix2_L), w-1);
					
//					float g_d =  expf( - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
                    float g_d = (1.0f - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
//                          g_d = g_d * g_d;
					float gradoutput_data_value = gradoutput[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n];
//					if(output[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n] < 0)
 //                           printf("Error g_d ==> %f \n",output[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n] );
                             if(output[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n] < threshhold)
                            {
                                //printf("Reliable gpu backward, under threshhold ==> %f\n",
                                 //   output[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n]);
                              continue;//to  skip its gradients
					        }
					//flow gradients
					atomicAdd( & gradinput3[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i],
							- gradoutput_data_value *  g_d * (n - alpha) / ( sigma_d * sigma_d) * 2.0f);
					atomicAdd( & gradinput3[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i],
							- gradoutput_data_value *   g_d *(m - beta)  / (  sigma_d * sigma_d) * 2.0f);
				}												
			}											
		}
	}	
	return ;

}

int ReliableWeightLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    			const int batch,

		//const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		//const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,
		
		const float* input3,   const float* output,
 
		const float* gradoutput,  float* gradinput3,
        float threshhold,
		float	sigma_d,     float tao_r , float  Prowindow

		)
{
	int error = -1;

	dim3 grid;
	dim3 block;


	//blockthread = 128;
	//the threadIdx.x is sheduled first, then threadIdx.y, threadIdx.z
	//the three channels are processsed in one kernel
	block  = dim3(BLOCKDIMX,BLOCKDIMY,1);
	grid = dim3( (w + BLOCKDIMX - 1)/ BLOCKDIMX, (h + BLOCKDIMY - 1) / BLOCKDIMY, batch);
    if(BLOCKDIMX != 32 || BLOCKDIMY != 16||DEBUG)
        printf("BLOCKDIMX revised to %d, BLOCKDIMY revised to %d \n", BLOCKDIMX,BLOCKDIMY);

//    cudaMemset((void*)gradinput1, 0, input1_b_stride * batch * sizeof(float));
//    cudaMemset((void*)gradinput2, 0, input2_b_stride * batch * sizeof(float));
//    cudaMemset((void*)gradinput3, 0, input3_b_stride * batch * sizeof(float));

	ReliableWeightLayer_gpu_backward_kernelfunc <<<grid,block,0, stream>>>(
			nElement, //to let the nummous
			w,h,
			//input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			//input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			//flow_weights_b_stride,flow_weights_c_stride,flow_weights_h_stride,flow_weights_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,


			input3, output,
			gradoutput, gradinput3,
			threshhold,
			sigma_d,      tao_r ,   Prowindow

			);

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("gpuerror in BilinearSampler.updateGradInput %s\n", cudaGetErrorString(err));
		//THError("aborting");
		return error;
	}

	error = 0;
	return error;		
}
//#ifdef __cplusplus
//	}
//#endif
