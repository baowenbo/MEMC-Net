#include <TH.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

// refer to :
// https://github.com/lmb-freiburg/flownet2/blob/master/src/caffe/layers/flow_warp_layer.cpp

#define min(a,b) ((a<b)?(a):(b))
#define max(a,b) ((a>b)?(a):(b))
#define abs(a) ((a>0) ?(a): (-a))

int SeparableConvFlowLayer_cpu_forward(
		THFloatTensor * input1,
		THFloatTensor * input2,
		THFloatTensor * input3,
		//THFloatTensor * output,
		THFloatTensor * flow_output
		)
		{
	int error = -1;

	int channel = input1->size[1]; //THTensor_size(state, input1, 1);
	if(channel!=3) return error;
	int batch = input1->size[0];//THCudaTensor_size(state, input1,0);
	if(input2->size[0] != batch) return error;
//	if(input2->size[1] != input2->size[1]) return error;


	int h = input1->size[2];//THCudaTensor_size(state,input1,2);
	int w = input1->size[3];//THCudaTensor_size(state,input1,3);
	if(input2->size[2] != h - input2->size[1] + 1) return error;// to add some checkpoint
	if(input2->size[3] != w - input2->size[1] + 1) return error;

	//float * input1_data = THFloatTensor_data(input1);
	float * input2_data = THFloatTensor_data(input2);
	float * input3_data = THFloatTensor_data(input3);
	//float * output_data = THFloatTensor_data(output);
	float * flow_output_data = THFloatTensor_data(flow_output);

	//int input1_b_stride = input1->stride[0];
	//int input1_c_stride = input1->stride[1];
	//int input1_h_stride = input1->stride[2];
	//int input1_w_stride = input1->stride[3];

	int input2_b_stride = input2->stride[0];
	int input2_c_stride = input2->stride[1];
	int input2_h_stride = input2->stride[2];
	int input2_w_stride = input2->stride[3];

	int input3_b_stride = input3->stride[0];
	int input3_c_stride = input3->stride[1];
	int input3_h_stride = input3->stride[2];
	int input3_w_stride = input3->stride[3];

//	int output_b_stride = output->stride[0];
//	int output_c_stride = output->stride[1];
//	int output_h_stride = output->stride[2];
//  int output_w_stride = output->stride[3];

	int flow_output_b_stride = flow_output->stride[0];
	int flow_output_c_stride = flow_output->stride[1];
	int flow_output_h_stride = flow_output->stride[2];
	int flow_output_w_stride = flow_output->stride[3];	
	//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);

	//TODO: do we need to assert the w_stride to be 1
	//if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
	if(input3_w_stride !=1) return error;
//	if(output_w_stride !=1) return error;
	if(flow_output_w_stride !=1) return error;

	if(input2_b_stride != input3_b_stride) return error;
	if(input2_c_stride != input3_c_stride) return error;



	//    printf("cpu: input1 %f, \t  output %f \n",input1_data[0],output_data[0]);
//	float fillvalue = 0.0f;//Default to be zero for the out of range position.
int batch_i;int h_i;int w_i;//int c_i;
int intFilterY;
int intFilterX ;
	for( batch_i = 0 ; batch_i < batch; batch_i++){
		for( h_i = 0; h_i < h - input2->size[1] + 1 ; h_i ++){
			for( w_i = 0; w_i < w - input2->size[1] + 1; w_i ++){

				float flow_y = 0.0f;
				float sum_weights = 0.0f;
				for (  intFilterY = 0; intFilterY < input2->size[1]; intFilterY += 1) {
					float temp2 = input2_data[batch_i * input2_b_stride + intFilterY * input2_c_stride + h_i * input2_h_stride + w_i ];
					flow_y += (float)(intFilterY) * temp2 ;
					sum_weights += 			temp2;
				}
				sum_weights = fabs(sum_weights);
				flow_y = flow_y / sum_weights - ((float)(input2->size[1])-1.0)/2.0;
				flow_output_data[batch_i * flow_output_b_stride + 1 * flow_output_c_stride+ h_i* flow_output_h_stride + w_i] = 
							sum_weights >0.0f ?  flow_y : -2000;
 
				float flow_x = 0.0f;
				float sum_weights_x = 0.0f;
				for (  intFilterX = 0; intFilterX < input2->size[1]; intFilterX += 1) {
					float temp3 = input3_data[batch_i * input3_b_stride + intFilterX * input3_c_stride + h_i * input3_h_stride + w_i ];
					flow_x += (float)(intFilterX)  * temp3;
					sum_weights_x += 		 temp3;
				}
				sum_weights_x = fabs(sum_weights_x);
				flow_x = flow_x / sum_weights_x - ((float)(input2->size[1])-1.0)/2.0;
				// what if the sum_weights is less than zeros.
				flow_output_data[batch_i * flow_output_b_stride + 0 * flow_output_c_stride + h_i* flow_output_h_stride + w_i] =
							sum_weights_x > 0.0f ? flow_x : -2000;
			}
		}
	}
	//    printf("cpu: input1 %f, \t  output %f \n",input1_data[0],output_data[0]);

	error = 0;
	return error;

}

//TODO: what is the correct order of the tensors in backward propagation although
int SeparableConvFlowLayer_cpu_backward(
		THFloatTensor * input1,
		THFloatTensor * input2,
		THFloatTensor * input3,
		THFloatTensor * gradflow_output,
		THFloatTensor * gradinput1,
		THFloatTensor * gradinput2,
		THFloatTensor * gradinput3
		)
		{
	int error = -1;


	int channel = input1->size[1]; //THCudaTensor_size(state, input1, 1);
	if(channel!=3) return error;
	int batch = input1->size[0];
	if(input2->size[0] != batch) return error;
//	if(input2->size[1] != input2->size[1]) return error;

	int h = input1->size[2];
	int w = input1->size[3];
	if(input2->size[2] != h - input2->size[1] + 1) return error;// to add some checkpoint
	if(input2->size[3] != w - input2->size[1] + 1) return error;

	float * input1_data = THFloatTensor_data(input1);
	float * input2_data = THFloatTensor_data(input2);
	float * input3_data = THFloatTensor_data(input3);
	float * gradflow_output_data = THFloatTensor_data(gradflow_output);
	float * gradinput1_data = THFloatTensor_data(gradinput1);
	float * gradinput2_data = THFloatTensor_data(gradinput2);
	float * gradinput3_data = THFloatTensor_data(gradinput3);


	//int input1_b_stride = input1->stride[0];
	//int input1_c_stride = input1->stride[1];
	//int input1_h_stride = input1->stride[2];
	//int input1_w_stride = input1->stride[3];

	int input2_b_stride = input2->stride[0];
	int input2_c_stride = input2->stride[1];
	int input2_h_stride = input2->stride[2];
	int input2_w_stride = input2->stride[3];

	int input3_b_stride = input3->stride[0];
	int input3_c_stride = input3->stride[1];
	int input3_h_stride = input3->stride[2];
	int input3_w_stride = input3->stride[3];

	int flow_output_b_stride = gradflow_output->stride[0];
	int flow_output_c_stride = gradflow_output->stride[1];
	int flow_output_h_stride = gradflow_output->stride[2];
	int flow_output_w_stride = gradflow_output->stride[3];
	//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);

	//TODO: do we need to assert the w_stride to be 1
	//if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
	if(input3_w_stride !=1) return error;
	if(flow_output_w_stride !=1) return error;

	if(input2_b_stride != input3_b_stride) return error;
	if(input2_c_stride != input3_c_stride) return error;


	//	printf("can i ");
	int batch_i;int h_i;int w_i;//int c_i;
	int intFilterY;
	int intFilterX;
	for( batch_i =0 ;batch_i < batch ; batch_i ++ ){
		for( h_i = 0 ; h_i < h - input2->size[1] + 1; h_i++){
			for( w_i = 0 ; w_i < w -input2->size[1] + 1; w_i ++){
				float flow_y = 0.0f;
				float sum_weights = 0.0f;
				for (  intFilterY = 0; intFilterY < input2->size[1]; intFilterY += 1) {
					float temp2 = input2_data[batch_i * input2_b_stride + intFilterY * input2_c_stride + h_i * input2_h_stride + w_i ];
					flow_y += (float)(intFilterY) * temp2 ;
					sum_weights += 			temp2;
				}
				//flow_y = flow_y / sum_weights - ((float)(input2->size[1])-1.0)/2.0;
				//flow_output_data[batch_i * flow_output_b_stride + 1 * flow_output_c_stride+ h_i* flow_output_h_stride + w_i] = 
				//		sum_weights >0 ?  flow_y : -2000;
				//float sign = sum_weights >0.0f ? 1.0f : -1.0f;
				//sum_weights = fabs(sum_weights);				
				if(fabs(sum_weights) >0.0f ){
					float gradflow_y = gradflow_output_data[batch_i * flow_output_b_stride + 1* flow_output_c_stride + 
										h_i * flow_output_h_stride + w_i ] ;					
					float offset = flow_y / ( sum_weights * sum_weights);
					for (  intFilterY = 0; intFilterY < input2->size[1]; intFilterY += 1) {
						gradinput2_data[batch_i * input2_b_stride + intFilterY * input2_c_stride  +  h_i * input2_h_stride + w_i ] =
									 gradflow_y *  ((float)(intFilterY) / sum_weights -  offset);
					}
				}
				
				
				
				float flow_x = 0.0f;
				float sum_weights_x = 0.0f;
				for (  intFilterX = 0; intFilterX < input2->size[1]; intFilterX += 1) {
					float temp3 = input3_data[batch_i * input3_b_stride + intFilterX * input3_c_stride + h_i * input3_h_stride + w_i ];
					flow_x += (float)(intFilterX)  * temp3;
					sum_weights_x += 		 temp3;
				}
				//flow_x = flow_x / sum_weights_x - ((float)(input2->size[1])-1.0)/2.0;
				//flow_output_data[batch_i * flow_output_b_stride + 0 * flow_output_c_stride + h_i* flow_output_h_stride + w_i] =
				//			sum_weights_x >0 ? flow_x : -2000;
				//float sign_x = sum_weights_x >0.0f ? 1.0f : -1.0f;
				//sum_weights_x = fabs(sum_weights_x);	
				if(fabs(sum_weights_x) > 0.0f ){
					 float gradflow_x = gradflow_output_data[batch_i * flow_output_b_stride + 0 * flow_output_c_stride + 
											h_i * flow_output_h_stride + w_i];
					float offset  = flow_x / (sum_weights_x * sum_weights_x);
					for (  intFilterX = 0; intFilterX < input2->size[1]; intFilterX += 1) {
						gradinput3_data[batch_i * input3_b_stride + intFilterX * input3_c_stride + h_i * input3_h_stride + w_i ] =
								 gradflow_x * ((float)(intFilterX) /sum_weights_x - offset);
					}
				}
					  
				
			}
		}
	}

	error = 0;
	return error;

}

int SeparableConvLayer_cpu_forward(
		THFloatTensor * input1,
		THFloatTensor * input2,
		THFloatTensor * input3,
		THFloatTensor * output
		)
		{
	int error = -1;

	int channel = input1->size[1]; //THTensor_size(state, input1, 1);
	if(channel!=3) return error;
	int batch = input1->size[0];//THCudaTensor_size(state, input1,0);
	if(input2->size[0] != batch) return error;
//	if(input2->size[1] != input2->size[1]) return error;


	int h = input1->size[2];//THCudaTensor_size(state,input1,2);
	int w = input1->size[3];//THCudaTensor_size(state,input1,3);
	if(input2->size[2] != h - input2->size[1] + 1) return error;// to add some checkpoint
	if(input2->size[3] != w - input2->size[1] + 1) return error;

	float * input1_data = THFloatTensor_data(input1);
	float * input2_data = THFloatTensor_data(input2);
	float * input3_data = THFloatTensor_data(input3);
	float * output_data = THFloatTensor_data(output);

	int input1_b_stride = input1->stride[0];
	int input1_c_stride = input1->stride[1];
	int input1_h_stride = input1->stride[2];
	int input1_w_stride = input1->stride[3];

	int input2_b_stride = input2->stride[0];
	int input2_c_stride = input2->stride[1];
	int input2_h_stride = input2->stride[2];
	int input2_w_stride = input2->stride[3];

	int input3_b_stride = input3->stride[0];
	int input3_c_stride = input3->stride[1];
	int input3_h_stride = input3->stride[2];
	int input3_w_stride = input3->stride[3];

	int output_b_stride = output->stride[0];
	int output_c_stride = output->stride[1];
	int output_h_stride = output->stride[2];
	int output_w_stride = output->stride[3];
	//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);

	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
	if(input3_w_stride !=1) return error;
	if(output_w_stride !=1) return error;

	if(input2_b_stride != input3_b_stride) return error;
	if(input2_c_stride != input3_c_stride) return error;



	//    printf("cpu: input1 %f, \t  output %f \n",input1_data[0],output_data[0]);
//	float fillvalue = 0.0f;//Default to be zero for the out of range position.
	int batch_i;int h_i;int w_i;int c_i;
 int intFilterY;
int intFilterX ;
	for( batch_i = 0 ; batch_i < batch; batch_i++){
		for( h_i = 0; h_i < h - input2->size[1] + 1 ; h_i ++){
			for( w_i = 0; w_i < w - input2->size[1] + 1; w_i ++){
				for ( c_i = 0 ; c_i < channel ; c_i ++){

					float out = 0.0f;
					for (  intFilterY = 0; intFilterY < input2->size[1]; intFilterY += 1) {
					for (  intFilterX = 0; intFilterX < input2->size[1]; intFilterX += 1) {
						float temp1 = input1_data[batch_i * input1_b_stride + c_i * input1_c_stride + (h_i + intFilterY )* input1_h_stride + (w_i + intFilterX)];
						float temp2 = input2_data[batch_i * input2_b_stride + intFilterY * input2_c_stride + h_i * input2_h_stride + w_i ];
						float temp3 = input3_data[batch_i * input3_b_stride + intFilterX * input3_c_stride + h_i * input3_h_stride + w_i ];
						out += temp1* temp2 * temp3;
					}
					}
					output_data[batch_i * output_b_stride + c_i* output_c_stride + h_i * output_h_stride + w_i ] = out;
				}
			}
		}
	}
	//    printf("cpu: input1 %f, \t  output %f \n",input1_data[0],output_data[0]);

	error = 0;
	return error;

}

//TODO: what is the correct order of the tensors in backward propagation although
int SeparableConvLayer_cpu_backward(
		THFloatTensor * input1,
		THFloatTensor * input2,
		THFloatTensor * input3,
		THFloatTensor * gradoutput,
		THFloatTensor * gradinput1,
		THFloatTensor * gradinput2,
		THFloatTensor * gradinput3
		)
		{
	int error = -1;


	int channel = input1->size[1]; //THCudaTensor_size(state, input1, 1);
	if(channel!=3) return error;
	int batch = input1->size[0];
	if(input2->size[0] != batch) return error;
//	if(input2->size[1] != input2->size[1]) return error;

	int h = input1->size[2];
	int w = input1->size[3];
	if(input2->size[2] != h - input2->size[1] + 1) return error;// to add some checkpoint
	if(input2->size[3] != w - input2->size[1] + 1) return error;

	float * input1_data = THFloatTensor_data(input1);
	float * input2_data = THFloatTensor_data(input2);
	float * input3_data = THFloatTensor_data(input3);
	float * gradoutput_data = THFloatTensor_data(gradoutput);
	float * gradinput1_data = THFloatTensor_data(gradinput1);
	float * gradinput2_data = THFloatTensor_data(gradinput2);
	float * gradinput3_data = THFloatTensor_data(gradinput3);


	int input1_b_stride = input1->stride[0];
	int input1_c_stride = input1->stride[1];
	int input1_h_stride = input1->stride[2];
	int input1_w_stride = input1->stride[3];

	int input2_b_stride = input2->stride[0];
	int input2_c_stride = input2->stride[1];
	int input2_h_stride = input2->stride[2];
	int input2_w_stride = input2->stride[3];

	int input3_b_stride = input3->stride[0];
	int input3_c_stride = input3->stride[1];
	int input3_h_stride = input3->stride[2];
	int input3_w_stride = input3->stride[3];

	int output_b_stride = gradoutput->stride[0];
	int output_c_stride = gradoutput->stride[1];
	int output_h_stride = gradoutput->stride[2];
	int output_w_stride = gradoutput->stride[3];
	//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);

	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
	if(input3_w_stride !=1) return error;
	if(output_w_stride !=1) return error;

	if(input2_b_stride != input3_b_stride) return error;
	if(input2_c_stride != input3_c_stride) return error;


	//	printf("can i ");
	int batch_i;int h_i;int w_i;int c_i;
int intFilterY;
int intFilterX ;
	for( batch_i =0 ;batch_i < batch ; batch_i ++ ){
		for( h_i = 0 ; h_i < h - input2->size[1] + 1; h_i++){
			for( w_i = 0 ; w_i < w -input2->size[1] + 1; w_i ++){

			for ( c_i = 0 ; c_i < channel ; c_i ++){

 					for (  intFilterY = 0; intFilterY < input2->size[1]; intFilterY += 1) {
					for (  intFilterX = 0; intFilterX < input2->size[1]; intFilterX += 1) {
						float temp1 = input1_data[batch_i * input1_b_stride + c_i * input1_c_stride + (h_i + intFilterY )* input1_h_stride + (w_i + intFilterX)];
						float temp2 = input2_data[batch_i * input2_b_stride + intFilterY * input2_c_stride + h_i * input2_h_stride + w_i ];
						float temp3 = input3_data[batch_i * input3_b_stride + intFilterX * input3_c_stride + h_i * input3_h_stride + w_i ];

						float gradout = gradoutput_data[batch_i * output_b_stride + c_i* output_c_stride + h_i * output_h_stride + w_i ];

						gradinput1_data[batch_i * input1_b_stride + c_i * input1_c_stride + (h_i + intFilterY )* input1_h_stride + (w_i + intFilterX)] +=
							gradout * temp2 * temp3;
						gradinput2_data[batch_i * input2_b_stride + intFilterY * input2_c_stride  +  h_i * input2_h_stride + w_i ] +=
							gradout * temp1 * temp3;
						gradinput3_data	[batch_i * input3_b_stride + intFilterX * input3_c_stride + h_i * input3_h_stride + w_i ] +=
							gradout * temp1 * temp2;
					}
					}
				}
			}
		}
	}

	error = 0;
	return error;

}

int InterpolationLayer_cpu_forward(
		THFloatTensor * input1,
		THFloatTensor * input2,
		THFloatTensor * output
		)
		{
	int error = -1;


	int channel = input1->size[1]; //THTensor_size(state, input1, 1);
	if(channel!=3) return error;
	int batch = input1->size[0];//THCudaTensor_size(state, input1,0);
	if(input2->size[0] != batch) return error;
	if(input2->size[1] != 2) return error;

	int h = input1->size[2];//THCudaTensor_size(state,input1,2);
	int w = input1->size[3];//THCudaTensor_size(state,input1,3);
	if(input2->size[2] != h) return error;// to add some checkpoint
	if(input2->size[3] != w) return error;

	float * input1_data =  THFloatTensor_data(input1);
	float * input2_data = THFloatTensor_data(input2);
	float * output_data = THFloatTensor_data(output);

	int input1_b_stride = input1->stride[0];
	int input1_c_stride = input1->stride[1];
	int input1_h_stride = input1->stride[2];
	int input1_w_stride = input1->stride[3];

	int input2_b_stride = input2->stride[0];
	int input2_c_stride = input2->stride[1];
	int input2_h_stride = input2->stride[2];
	int input2_w_stride = input2->stride[3];
	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
	if(input1_b_stride != output->stride[0]) return error;
	if(input1_c_stride != output->stride[1]) return error;


	float fillvalue = 0.0f;//Default to be zero for the out of range position.
	int batch_i;int h_i;int w_i;int c_i;
	for( batch_i = 0 ; batch_i < batch; batch_i++){
		int off = batch_i * input1_b_stride;
		for( h_i = 0; h_i < h ; h_i ++){
			for( w_i = 0; w_i < w; w_i ++){

				//read the opticalflow
				float fx = input2_data[batch_i * input2_b_stride + 0 * input2_c_stride + h_i * input2_h_stride + w_i];
				float fy = input2_data[batch_i * input2_b_stride + 1 * input2_c_stride + h_i * input2_h_stride + w_i];

				//get the destination position
				float x2 = (float)(w_i) + fx;
				float y2 = (float)(h_i) + fy;

				if(x2 >= 0.0f && y2 >=0.0f && x2 < (float)w && y2  < (float)h){
					int ix2_L = (int)(x2);
					int iy2_T = (int)(y2);
					int ix2_R = min(ix2_L+1, w - 1);
					int iy2_B = min(iy2_T+1, h - 1);

					float alpha = x2 - ix2_L;
					float beta = y2  - iy2_T;

					for ( c_i = 0 ; c_i  < channel; c_i++){
						float TL = input1_data[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_L];
						float TR = input1_data[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_R];
						float BL = input1_data[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_L];
						float BR = input1_data[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_R];

						output_data[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i] =
							(1-alpha)*(1-beta)*TL +
							alpha*(1-beta)*TR +
							(1-alpha)*beta*BL +
							alpha*beta*BR;
					}


				}else{
					//the warping data is out of range, we fill it with zeros
					for( c_i = 0 ;  c_i < channel; c_i ++){
						output_data[off +c_i *input1_c_stride + h_i* input1_h_stride + w_i] = fillvalue;
					}
				}
			}
		}
	}

	error = 0;
	return error;

}

//TODO: what is the correct order of the tensors in backward propagation although
int InterpolationLayer_cpu_backward(
		THFloatTensor * input1,
		THFloatTensor * input2,
		THFloatTensor * gradoutput,
		THFloatTensor * gradinput1,
		THFloatTensor * gradinput2
		)
		{
	int error = -1;


	int channel = input1->size[1]; //THCudaTensor_size(state, input1, 1);
	if(channel!=3) return error;
	int batch = input1->size[0];
	if(input2->size[0] != batch) return error;
	if(input2->size[1] != 2) return error;

	int h = input1->size[2];
	int w = input1->size[3];
	if(input2->size[2] != h) return error;// to add some checkpoint
	if(input2->size[3] != w) return error;

	float * input1_data = THFloatTensor_data(input1);
	float * input2_data = THFloatTensor_data(input2);
	float * gradoutput_data = THFloatTensor_data(gradoutput);
	float * gradinput1_data = THFloatTensor_data(gradinput1);
	float * gradinput2_data = THFloatTensor_data(gradinput2);


	int input1_b_stride = input1->stride[0];
	int input1_c_stride = input1->stride[1];
	int input1_h_stride = input1->stride[2];
	int input1_w_stride = input1->stride[3];

	int input2_b_stride = input2->stride[0];
	int input2_c_stride = input2->stride[1];
	int input2_h_stride = input2->stride[2];
	int input2_w_stride = input2->stride[3];
	//		printf("can id ");
	//    printf("%d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);
	if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
	//	printf("can  ");

	//    int i ;
	//clear the gradients of the input1
	//	for ( i = 0 ; i < batch * input1_b_stride; i ++){ gradinput1_data[i] = 0;}
	//clear the gradients of the input2
	//	for( i = 0 ; i < batch * input2_b_stride ; i ++){ gradinput2_data[i] = 0;}

	//	printf("can i ");
	int batch_i;int h_i;int w_i;int c_i;

	for( batch_i =0 ;batch_i < batch ; batch_i ++ ){
		//	printf("%d\n",batch_i);
		int off = batch_i * input1_b_stride;

		for( h_i = 0 ; h_i< h; h_i++){
			for( w_i = 0 ; w_i < w; w_i ++){

				//read the opticalflow
				float fx = input2_data[batch_i * input2_b_stride + 0 * input2_c_stride + h_i * input2_h_stride + w_i];
				float fy = input2_data[batch_i * input2_b_stride + 1 * input2_c_stride + h_i * input2_h_stride + w_i];

				//get the destination position
				float x2 = (float)(w_i) + fx;
				float y2 = (float)(h_i) + fy;

				if(x2 >=  0.0f  && y2 >= 0.0f  && x2 < (float)w && y2 < (float) h){
					int ix2_L = (int)(x2);
					int iy2_T = (int)(y2);
					int ix2_R = min(ix2_L+1, w - 1);
					int iy2_B = min(iy2_T+1, h - 1);

					float alpha = x2 - ix2_L;
					float beta = y2  - iy2_T;

					for ( c_i = 0 ; c_i  < channel; c_i++){

						float gradoutput_value = gradoutput_data[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i];
						gradinput1_data[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_L] += gradoutput_value * (1-alpha)*(1-beta);
						gradinput1_data[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_R] += gradoutput_value * alpha*(1-beta);
						gradinput1_data[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_L] += gradoutput_value * (1-alpha)*beta;
						gradinput1_data[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_R] += gradoutput_value *  alpha*beta;

					}


					float gamma  = iy2_B - y2;

					float bot_diff = 0 ;
					for( c_i = 0 ; c_i < channel;c_i ++ ){
						float temp = 0.0f;
						temp += gamma * (input1_data[off+ c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_R] -
								input1_data[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_L]);
						temp += (1-gamma) *(input1_data[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_R] -
								input1_data[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_L]);

						float warped_diff_value = gradoutput_data[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i];

						bot_diff += warped_diff_value * temp;
					}
					//the gradients of the x direction/ horizontal direction
					gradinput2_data[batch_i * input2_b_stride + 0 * input2_c_stride + h_i * input2_h_stride + w_i] = bot_diff;


					gamma = ix2_R -x2;
					bot_diff = 0;

					for ( c_i = 0 ; c_i < channel ; c_i++){
						float temp = 0.0f;
						temp += gamma * (input1_data[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_L] -
								input1_data[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_L ]);
						temp += (1-gamma) * (  input1_data[off + c_i * input1_c_stride+ iy2_B * input1_h_stride + ix2_R] -
								input1_data[off +c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_R]);

						float warped_diff_value = gradoutput_data[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i];

						bot_diff += warped_diff_value * temp;

					}
					gradinput2_data[batch_i * input2_b_stride + 1 * input2_c_stride + h_i * input2_h_stride + w_i] = bot_diff;

				}
			}
		}
	}
	//    printf("Finish \n");

	error = 0;
	return error;

}

int InterpolationChLayer_cpu_forward(
		THFloatTensor * input1,
		THFloatTensor * input2,
		THFloatTensor * output
		)
		{
	int error = -1;


	int channel = input1->size[1]; //THTensor_size(state, input1, 1);
//	if(channel!=3) return error;
	int batch = input1->size[0];//THCudaTensor_size(state, input1,0);
	if(input2->size[0] != batch) return error;
	if(input2->size[1] != 2) return error;

	int h = input1->size[2];//THCudaTensor_size(state,input1,2);
	int w = input1->size[3];//THCudaTensor_size(state,input1,3);
	if(input2->size[2] != h) return error;// to add some checkpoint
	if(input2->size[3] != w) return error;

	float * input1_data =  THFloatTensor_data(input1);
	float * input2_data = THFloatTensor_data(input2);
	float * output_data = THFloatTensor_data(output);

	int input1_b_stride = input1->stride[0];
	int input1_c_stride = input1->stride[1];
	int input1_h_stride = input1->stride[2];
	int input1_w_stride = input1->stride[3];

	int input2_b_stride = input2->stride[0];
	int input2_c_stride = input2->stride[1];
	int input2_h_stride = input2->stride[2];
	int input2_w_stride = input2->stride[3];
	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
	if(input1_b_stride != output->stride[0]) return error;
	if(input1_c_stride != output->stride[1]) return error;


	float fillvalue = 0.0f;//Default to be zero for the out of range position.
	int batch_i;int h_i;int w_i;int c_i;
	for( batch_i = 0 ; batch_i < batch; batch_i++){
		int off = batch_i * input1_b_stride;
		for( h_i = 0; h_i < h ; h_i ++){
			for( w_i = 0; w_i < w; w_i ++){

				//read the opticalflow
				float fx = input2_data[batch_i * input2_b_stride + 0 * input2_c_stride + h_i * input2_h_stride + w_i];
				float fy = input2_data[batch_i * input2_b_stride + 1 * input2_c_stride + h_i * input2_h_stride + w_i];

				//get the destination position
				float x2 = (float)(w_i) + fx;
				float y2 = (float)(h_i) + fy;

				if(x2 >= 0.0f && y2 >=0.0f && x2 < (float)w && y2  < (float)h){
					int ix2_L = (int)(x2);
					int iy2_T = (int)(y2);
					int ix2_R = min(ix2_L+1, w - 1);
					int iy2_B = min(iy2_T+1, h - 1);

					float alpha = x2 - ix2_L;
					float beta = y2  - iy2_T;

					for ( c_i = 0 ; c_i  < channel; c_i++){
						float TL = input1_data[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_L];
						float TR = input1_data[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_R];
						float BL = input1_data[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_L];
						float BR = input1_data[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_R];

						output_data[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i] =
							(1-alpha)*(1-beta)*TL +
							alpha*(1-beta)*TR +
							(1-alpha)*beta*BL +
							alpha*beta*BR;
					}
				}else{
					//the warping data is out of range, we fill it with zeros
					for( c_i = 0 ;  c_i < channel; c_i ++){
						output_data[off +c_i *input1_c_stride + h_i* input1_h_stride + w_i] = fillvalue;
					}
				}
			}
		}
	}

	error = 0;
	return error;

}

//TODO: what is the correct order of the tensors in backward propagation although
int InterpolationChLayer_cpu_backward(
		THFloatTensor * input1,
		THFloatTensor * input2,
		THFloatTensor * gradoutput,
		THFloatTensor * gradinput1,
		THFloatTensor * gradinput2
		)
		{
	int error = -1;


	int channel = input1->size[1]; //THCudaTensor_size(state, input1, 1);
//	if(channel!=3) return error;
	int batch = input1->size[0];
	if(input2->size[0] != batch) return error;
	if(input2->size[1] != 2) return error;

	int h = input1->size[2];
	int w = input1->size[3];
	if(input2->size[2] != h) return error;// to add some checkpoint
	if(input2->size[3] != w) return error;

	float * input1_data = THFloatTensor_data(input1);
	float * input2_data = THFloatTensor_data(input2);
	float * gradoutput_data = THFloatTensor_data(gradoutput);
	float * gradinput1_data = THFloatTensor_data(gradinput1);
	float * gradinput2_data = THFloatTensor_data(gradinput2);


	int input1_b_stride = input1->stride[0];
	int input1_c_stride = input1->stride[1];
	int input1_h_stride = input1->stride[2];
	int input1_w_stride = input1->stride[3];

	int input2_b_stride = input2->stride[0];
	int input2_c_stride = input2->stride[1];
	int input2_h_stride = input2->stride[2];
	int input2_w_stride = input2->stride[3];
	//		printf("can id ");
	//    printf("%d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);
	if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
	//	printf("can  ");
//	if(input1_b_stride != output->stride[0]) return error;
//	if(input1_c_stride != output->stride[1]) return error;
	//    int i ;
	//clear the gradients of the input1
	//	for ( i = 0 ; i < batch * input1_b_stride; i ++){ gradinput1_data[i] = 0;}
	//clear the gradients of the input2
	//	for( i = 0 ; i < batch * input2_b_stride ; i ++){ gradinput2_data[i] = 0;}
	//	printf("can i ");
	int batch_i;int h_i;int w_i;int c_i;

	for( batch_i =0 ;batch_i < batch ; batch_i ++ ){
		//	printf("%d\n",batch_i);
		int off = batch_i * input1_b_stride;

		for( h_i = 0 ; h_i< h; h_i++){
			for( w_i = 0 ; w_i < w; w_i ++){

				//read the opticalflow
				float fx = input2_data[batch_i * input2_b_stride + 0 * input2_c_stride + h_i * input2_h_stride + w_i];
				float fy = input2_data[batch_i * input2_b_stride + 1 * input2_c_stride + h_i * input2_h_stride + w_i];

				//get the destination position
				float x2 = (float)(w_i) + fx;
				float y2 = (float)(h_i) + fy;

				if(x2 >=  0.0f  && y2 >= 0.0f  && x2 < (float)w && y2 < (float) h){
					int ix2_L = (int)(x2);
					int iy2_T = (int)(y2);
					int ix2_R = min(ix2_L+1, w - 1);
					int iy2_B = min(iy2_T+1, h - 1);

					float alpha = x2 - ix2_L;
					float beta = y2  - iy2_T;

					for ( c_i = 0 ; c_i  < channel; c_i++){

						float gradoutput_value = gradoutput_data[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i];
						gradinput1_data[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_L] += gradoutput_value * (1-alpha)*(1-beta);
						gradinput1_data[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_R] += gradoutput_value * alpha*(1-beta);
						gradinput1_data[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_L] += gradoutput_value * (1-alpha)*beta;
						gradinput1_data[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_R] += gradoutput_value *  alpha*beta;

					}


					float gamma  = iy2_B - y2;

					float bot_diff = 0 ;
					for( c_i = 0 ; c_i < channel;c_i ++ ){
						float temp = 0.0f;
						temp += gamma * (input1_data[off+ c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_R] -
								input1_data[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_L]);
						temp += (1-gamma) *(input1_data[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_R] -
								input1_data[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_L]);

						float warped_diff_value = gradoutput_data[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i];

						bot_diff += warped_diff_value * temp;
					}
					//the gradients of the x direction/ horizontal direction
					gradinput2_data[batch_i * input2_b_stride + 0 * input2_c_stride + h_i * input2_h_stride + w_i] = bot_diff;


					gamma = ix2_R -x2;
					bot_diff = 0;

					for ( c_i = 0 ; c_i < channel ; c_i++){
						float temp = 0.0f;
						temp += gamma * (input1_data[off + c_i * input1_c_stride + iy2_B * input1_h_stride + ix2_L] -
								input1_data[off + c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_L ]);
						temp += (1-gamma) * (  input1_data[off + c_i * input1_c_stride+ iy2_B * input1_h_stride + ix2_R] -
								input1_data[off +c_i * input1_c_stride + iy2_T * input1_h_stride + ix2_R]);

						float warped_diff_value = gradoutput_data[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i];

						bot_diff += warped_diff_value * temp;

					}
					gradinput2_data[batch_i * input2_b_stride + 1 * input2_c_stride + h_i * input2_h_stride + w_i] = bot_diff;

				}
			}
		}
	}
	//    printf("Finish \n");

	error = 0;
	return error;

}
#include <TH.h>
#include <stdbool.h>
#include <stdio.h>

// refer to :
// https://github.com/lmb-freiburg/flownet2/blob/master/src/caffe/layers/flow_warp_layer.cpp

#define min(a,b) ((a<b)?(a):(b))
#define max(a,b) ((a>b)?(a):(b))


int FilterInterpolationLayer_cpu_forward(
		THFloatTensor * input1,
		THFloatTensor * input2,
		THFloatTensor * input3,
		THFloatTensor * output
		)
		{
	int error = -1;

	int channel = input1->size[1]; //THTensor_size(state, input1, 1);
	//if(channel!=3) return error;
	int batch = input1->size[0];//THCudaTensor_size(state, input1,0);
	if(input2->size[0] != batch) return error;
	if(input2->size[1] != 2) return error;

	int h = input1->size[2];//THCudaTensor_size(state,input1,2);
	int w = input1->size[3];//THCudaTensor_size(state,input1,3);
	if(input2->size[2] != h) return error;// to add some checkpoint
	if(input2->size[3] != w) return error;

	int filter_size2 = input3->size[1];
	int filter_size = (int)sqrt((float)filter_size2);
	//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));

	float * input1_data = THFloatTensor_data(input1);
	float * input2_data = THFloatTensor_data(input2);
	float * input3_data = THFloatTensor_data(input3);
	float * output_data = THFloatTensor_data(output);

	int input1_b_stride = input1->stride[0];
	int input1_c_stride = input1->stride[1];
	int input1_h_stride = input1->stride[2];
	int input1_w_stride = input1->stride[3];

	int input2_b_stride = input2->stride[0];
	int input2_c_stride = input2->stride[1];
	int input2_h_stride = input2->stride[2];
	int input2_w_stride = input2->stride[3];

	int input3_b_stride = input3->stride[0];
	int input3_c_stride = input3->stride[1];
	int input3_h_stride = input3->stride[2];
	int input3_w_stride = input3->stride[3];
	//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);

	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
	if(input3_w_stride !=1) return error;
	if(input1_b_stride != output->stride[0]) return error;
	if(input1_c_stride != output->stride[1]) return error;



	//    printf("cpu: input1 %f, \t  output %f \n",input1_data[0],output_data[0]);
//	float fillvalue = 0.0f;//Default to be zero for the out of range position.
	int batch_i;int h_i;int w_i;int c_i;
	int filter_i,filter_j;

	for( batch_i = 0 ; batch_i < batch; batch_i++){
		int off = batch_i * input1_b_stride;
		for( h_i = 0; h_i < h ; h_i ++){
			for( w_i = 0; w_i < w; w_i ++){

				//read the opticalflow
				float fx = input2_data[batch_i * input2_b_stride + 0 * input2_c_stride + h_i * input2_h_stride + w_i];
				float fy = input2_data[batch_i * input2_b_stride + 1 * input2_c_stride + h_i * input2_h_stride + w_i];

				//get the destination position
				float x2 = (float)(w_i) + fx;
				float y2 = (float)(h_i) + fy;

				//Guarrantee that the center position is in-border.
				if(x2 >= 0.0f && y2 >=0.0f && x2 <=  (float) (w -1) && y2  <= (float) (h-1)
				        && fabs(fx) < (float)(w)/2.0f && fabs(fy) < (float)(h)/2.0f){

					//TODO: what if ix2_L goes out of the border, then
					int ix2_L = (int)(x2) + 1 - (int)(filter_size/2); // the very left coordinate though it may goes out border.
					int iy2_T = (int)(y2) + 1 - (int)(filter_size/2); // the very top coordinate
					int ix2_R = ix2_L + filter_size; // the very right coordinate
					int iy2_B = iy2_T + filter_size; // the very bottom coordinate

					float alpha = x2 - (int)(x2);
					float beta = y2  - (int)(y2);

					//TODO: here is a bug that if the iy2_B or ix2_R gets out of the border, than there is no enough pixels to warp the target one.
					for ( c_i = 0 ; c_i < channel; c_i ++) {
					// the left top part
//					float temp = 0.0f;

					float TL = 0.0f;
					for (filter_j = iy2_T; filter_j <= (int) (y2); filter_j ++ ){
                        int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
					for (filter_i = ix2_L; filter_i <= (int) (x2); filter_i ++ ){
                        int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                        TL += input1_data [off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
                            input3_data [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];
					}
					}

					float TR = 0.0f;
					for (filter_j = iy2_T; filter_j <= (int) (y2); filter_j ++ ){
                        int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
					for (filter_i =  (int) (x2) + 1 ; filter_i < ix2_R; filter_i ++ ){
                        int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                        TR += input1_data [off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
                            input3_data [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];
					}
					}

					float BL = 0.0f;
					for (filter_j = (int) (y2) + 1; filter_j < iy2_B; filter_j ++ ){
                        int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
					for (filter_i = ix2_L; filter_i <= (int) (x2); filter_i ++ ){
                        int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                        BL += input1_data [off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
                            input3_data [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];
					}
					}

					float BR = 0.0f;
					for (filter_j = (int) (y2) + 1; filter_j < iy2_B; filter_j ++ ){
                        int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
					for (filter_i = (int) (x2) + 1; filter_i < ix2_R; filter_i ++ ){
                        int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                        BR += input1_data [off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
                            input3_data [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];
					}
					}

                    output_data [off + c_i * input1_c_stride + h_i * input1_h_stride + w_i] =
                            (1-alpha)*(1-beta)*TL +
							alpha*(1-beta)*TR +
							(1-alpha)*beta*BL +
							alpha*beta*BR;
					}

//					for ( c_i = 0 ; c_i  < channel; c_i++){
//						for (filter_j = iy2_T ; filter_j < iy2_B; filter_j++){
//							int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
//
//							for(filter_i = ix2_L; filter_i < ix2_R; filter_i++){
//								int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
//
//								output_data [off + c_i * input1_c_stride + h_i * input1_h_stride + w_i] +=
//									input1_data [off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
//									input3_data [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i] *
////									exp( - (fabs((float)(filter_j) - y2 ) + fabs((float)(filter_i)  - x2))/(float)(filter_size) ) ; // the distance weight.
//									exp( - (fabs((float)(filter_j) - y2 ) + fabs((float)(filter_i)  - x2)) ) ; // the distance weight.
//
////																if(w_i == 141 && h_i == 316 && c_i == 0 ){
////printf("gpu: %f, %f,%f,%f\n",input1_data[off + c_i *  input1_c_stride +  _filter_j * input1_h_stride + _filter_i ] ,
////input3_data [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i],
////exp( -(fabs((float) filter_j - y2) + fabs((float) filter_i - x2)) / (float)(filter_size)),
////output_data[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i ]
//// );
////}
//							}
//						}
//					}

				}else{
					//the warping data is out of range, we fill it with zeros
					for( c_i = 0 ;  c_i < channel; c_i ++){
						output_data[off + c_i *input1_c_stride + h_i* input1_h_stride + w_i] = input1_data[off + c_i* input1_c_stride+ h_i * input1_h_stride + w_i];
					}
				}
			}
		}
	}
	//    printf("cpu: input1 %f, \t  output %f \n",input1_data[0],output_data[0]);

	error = 0;
	return error;

}

//TODO: what is the correct order of the tensors in backward propagation although
int FilterInterpolationLayer_cpu_backward(
		THFloatTensor * input1,
		THFloatTensor * input2,
		THFloatTensor * input3,
		THFloatTensor * gradoutput,
		THFloatTensor * gradinput1,
		THFloatTensor * gradinput2,
		THFloatTensor * gradinput3
		)
		{
	int error = -1;
 

	int channel = input1->size[1]; //THCudaTensor_size(state, input1, 1);
	//if(channel!=3) return error;
	int batch = input1->size[0];
	if(input2->size[0] != batch) return error;
	if(input2->size[1] != 2) return error;

	int h = input1->size[2];
	int w = input1->size[3];
	if(input2->size[2] != h) return error;// to add some checkpoint
	if(input2->size[3] != w) return error;
	int filter_size2 = input3->size[1];
	int filter_size = (int)sqrt((float)filter_size2);
	//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));

	float * input1_data = THFloatTensor_data(input1);
	float * input2_data = THFloatTensor_data(input2);
	float * input3_data = THFloatTensor_data(input3);
	float * gradoutput_data = THFloatTensor_data(gradoutput);
	float * gradinput1_data = THFloatTensor_data(gradinput1);
	float * gradinput2_data = THFloatTensor_data(gradinput2);
	float * gradinput3_data = THFloatTensor_data(gradinput3);


	int input1_b_stride = input1->stride[0];
	int input1_c_stride = input1->stride[1];
	int input1_h_stride = input1->stride[2];
	int input1_w_stride = input1->stride[3];

	int input2_b_stride = input2->stride[0];
	int input2_c_stride = input2->stride[1];
	int input2_h_stride = input2->stride[2];
	int input2_w_stride = input2->stride[3];

	int input3_b_stride = input3->stride[0];
	int input3_c_stride = input3->stride[1];
	int input3_h_stride = input3->stride[2];
	int input3_w_stride = input3->stride[3];
	//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);

	//		printf("can id ");
	//    printf("%d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);
	if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
	if(input3_w_stride !=1) return error;
	//	printf("can  ");

//	    int i ;
//	//clear the gradients of the input1
//		for ( i = 0 ; i < batch * input1_b_stride; i ++){ gradinput1_data[i] = 0;}
//		for( i = 0 ; i < batch * input2_b_stride ; i ++){ gradinput2_data[i] = 0;}
//		for( i = 0 ; i < batch * input3_b_stride ; i ++){ gradinput3_data[i] = 0;}

	//	printf("can i ");
	int batch_i;int h_i;int w_i;int c_i;
	int filter_i,filter_j;

	for( batch_i =0 ;batch_i < batch ; batch_i ++ ){
		//	printf("%d\n",batch_i);
		int off = batch_i * input1_b_stride;

		for( h_i = 0 ; h_i < h; h_i++){
			for( w_i = 0 ; w_i < w; w_i ++){

				//read the opticalflow
				float fx = input2_data[batch_i * input2_b_stride + 0 * input2_c_stride + h_i * input2_h_stride + w_i];
				float fy = input2_data[batch_i * input2_b_stride + 1 * input2_c_stride + h_i * input2_h_stride + w_i];

				//get the destination position
				float x2 = (float)(w_i) + fx;
				float y2 = (float)(h_i) + fy;

				if(x2 >=  0.0f  && y2 >= 0.0f  && x2 <= (float)(w -1) && y2 <= (float) (h -1)
				        && fabs(fx) < (float)(w)/2.0f && fabs(fy) < (float)(h)/2.0f){

//                if (w_i == w/2 && h_i == h/2 && batch_i == 0 ){
//                    printf("%f,%f,%f,%f\n",fx ,fy,x2,y2);
//                }

					int ix2_L = (int)(x2) + 1 - (int) (filter_size/2);
					int iy2_T = (int)(y2) + 1 - (int) (filter_size/2);
					int ix2_R = ix2_L + filter_size ;
					int iy2_B = iy2_T + filter_size ;

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
					for ( c_i = 0 ; c_i  < channel; c_i++){

						float gradoutput_value = gradoutput_data[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i];

                        float TL_grad = gradoutput_value * (1-alpha) * ( 1-beta);
                        for (filter_j = iy2_T; filter_j <= (int) (y2); filter_j ++ ){
                            int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                        for (filter_i = ix2_L; filter_i <= (int) (x2); filter_i ++ ){
                            int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                                gradinput1_data[off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] +=
                                TL_grad *
                                input3_data [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];

                                gradinput3_data[ batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i] +=
                                TL_grad *
                                input1_data[off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i ];
                            }
                        }


                        float TR_grad = gradoutput_value * alpha * (1-beta);
                        for (filter_j = iy2_T; filter_j <= (int) (y2); filter_j ++ ){
                            int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                        for (filter_i =  (int) (x2) + 1 ; filter_i < ix2_R; filter_i ++ ){
                            int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                            gradinput1_data[off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] +=
                                    TR_grad * input3_data [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) *
                                                                            input3_c_stride + h_i * input3_h_stride + w_i];

                            gradinput3_data[ batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) *
                                                                            input3_c_stride + h_i * input3_h_stride + w_i] +=
                                    TR_grad * input1_data[off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i ];
                                    }
                        }

                        float BL_grad =gradoutput_value * (1-alpha) * beta;
                        for (filter_j = (int) (y2) + 1; filter_j < iy2_B; filter_j ++ ){
                            int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                        for (filter_i = ix2_L; filter_i <= (int) (x2); filter_i ++ ){
                            int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                             gradinput1_data[off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] +=
                                    BL_grad*
                                    input3_data [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];

                            gradinput3_data[ batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i] +=
                                    BL_grad *
                                    input1_data[off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i ];
                            }
                        }

                        float BR_grad = gradoutput_value * alpha * beta;
                        for (filter_j = (int) (y2) + 1; filter_j < iy2_B; filter_j ++ ){
                            int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                        for (filter_i = (int) (x2) + 1; filter_i < ix2_R; filter_i ++ ){
                            int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                                gradinput1_data[off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] +=
                                    BR_grad*
                                    input3_data [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];

                                gradinput3_data[ batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i] +=
                                    BR_grad *
                                    input1_data[off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i ];
                        }
                        }

//						for (filter_j = iy2_T; filter_j < iy2_B ; filter_j ++ ){
//							int _filter_j = min(max(0, filter_j),  h - 1);
//							for(filter_i = ix2_L; filter_i< ix2_R ; filter_i++){
//								int _filter_i = min(max(0,filter_i), w - 1);
//
//								gradinput1_data[off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] +=
//									gradoutput_value *
//									input3_data [batch_i * input3_b_stride +((filter_j - iy2_T) * filter_size + (filter_i-ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i] *
////									exp( -(fabs((float)filter_j - y2) + fabs((float)filter_i - x2))/(float)filter_size);
//									exp( -(fabs((float)filter_j - y2) + fabs((float)filter_i - x2)));
//
//							}
//						}
					}

					/***
					  Step 2: calculate the gradients for input2, i.e., the optical flow,
					  STEP 2.1: for the x/horizonotal direction.
					 ***/
                    float gamma  = 1.0f - beta; // iy2_B - y2;
					float bot_diff = 0.0f ;
					for( c_i = 0 ; c_i < channel;c_i ++ ){

						float gradoutput_value = gradoutput_data[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i];

                        float TL = 0.0f;
                        for (filter_j = iy2_T; filter_j <= (int) (y2); filter_j ++ ){
                            int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                        for (filter_i = ix2_L; filter_i <= (int) (x2); filter_i ++ ){
                            int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                            TL += input1_data [off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
                                input3_data [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];
                        }
                        }

                        float TR = 0.0f;
                        for (filter_j = iy2_T; filter_j <= (int) (y2); filter_j ++ ){
                            int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                        for (filter_i =  (int) (x2) + 1 ; filter_i < ix2_R; filter_i ++ ){
                            int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                            TR += input1_data [off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
                                input3_data [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];
                        }
                        }

                        float BL = 0.0f;
                        for (filter_j = (int) (y2) + 1; filter_j < iy2_B; filter_j ++ ){
                            int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                        for (filter_i = ix2_L; filter_i <= (int) (x2); filter_i ++ ){
                            int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                            BL += input1_data [off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
                                input3_data [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];
                        }
                        }

                        float BR = 0.0f;
                        for (filter_j = (int) (y2) + 1; filter_j < iy2_B; filter_j ++ ){
                            int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                        for (filter_i = (int) (x2) + 1; filter_i < ix2_R; filter_i ++ ){
                            int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                            BR += input1_data [off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
                                input3_data [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];
                        }
                        }

						float temp = 0.0f;
						temp += gamma * (TR - TL);
						temp += (1.0f-gamma) * (BR - BL);
						bot_diff += gradoutput_value * temp;
//						for (filter_j = iy2_T; filter_j < iy2_B ; filter_j ++ ){
//							int _filter_j = min(max(0, filter_j),  h - 1);
//							for(filter_i = ix2_L; filter_i< ix2_R ; filter_i++){
//								int _filter_i = min(max(0,filter_i), w - 1);
//
//								bot_diff +=
//									gradoutput_value *
//									input1_data[off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
//									input3_data [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i] *
////									exp( - (fabs((float) filter_j - y2)+fabs((float)filter_i - x2))/(float)filter_size ) *
////									((float)filter_i > x2 ? 1.0f: -1.0f)/(float)filter_size;
//	                                exp( - (fabs((float) filter_j - y2)+fabs((float)filter_i - x2)) ) *
//									((float)filter_i > x2 ? 1.0f: -1.0f);
//								//TODO: the problem is at the exact integer position i.e. filter_i == x2, in which is the gradient descent direction ???
//							}
//						}
					}
					//the gradients of the x direction/ horizontal direction
					gradinput2_data[batch_i * input2_b_stride + 0 * input2_c_stride + h_i * input2_h_stride + w_i] = bot_diff;

					/***
					  STEP 2.2: for the x/horizonotal direction.
					 ***/
                    gamma =  1.0f - alpha; //ix2_R -x2;
					bot_diff = 0.0f ;
					for ( c_i = 0 ; c_i < channel ; c_i++){
						float gradoutput_value = gradoutput_data[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i];


                        float TL = 0.0f;
                        for (filter_j = iy2_T; filter_j <= (int) (y2); filter_j ++ ){
                            int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                        for (filter_i = ix2_L; filter_i <= (int) (x2); filter_i ++ ){
                            int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                            TL += input1_data [off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
                                input3_data [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];
                        }
                        }

                        float TR = 0.0f;
                        for (filter_j = iy2_T; filter_j <= (int) (y2); filter_j ++ ){
                            int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                        for (filter_i =  (int) (x2) + 1 ; filter_i < ix2_R; filter_i ++ ){
                            int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                            TR += input1_data [off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
                                input3_data [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];
                        }
                        }

                        float BL = 0.0f;
                        for (filter_j = (int) (y2) + 1; filter_j < iy2_B; filter_j ++ ){
                            int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                        for (filter_i = ix2_L; filter_i <= (int) (x2); filter_i ++ ){
                            int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                            BL += input1_data [off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
                                input3_data [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];
                        }
                        }

                        float BR = 0.0f;
                        for (filter_j = (int) (y2) + 1; filter_j < iy2_B; filter_j ++ ){
                            int _filter_j = min(max(0, filter_j),h - 1); // only used for input1
                        for (filter_i = (int) (x2) + 1; filter_i < ix2_R; filter_i ++ ){
                            int _filter_i = min(max(0, filter_i),w - 1);// only used for input1
                            BR += input1_data [off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
                                input3_data [batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i];
                        }
                        }

	                    float temp = 0.0f;
						temp += gamma * (BL - TL);
						temp += (1.0f -gamma) * ( BR - 	TR);
						bot_diff += gradoutput_value * temp;

//						for (filter_j = iy2_T; filter_j < iy2_B ; filter_j ++ ){
//							int _filter_j = min(max(0, filter_j),  h - 1);
//							for(filter_i = ix2_L; filter_i< ix2_R ; filter_i++){
//								int _filter_i = min(max(0,filter_i), w - 1);
//
//								bot_diff+=
//									gradoutput_value *
//									input1_data[off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i] *
//									input3_data [batch_i * input3_b_stride + ((filter_j - iy2_T ) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i] *
////									exp( - (fabs((float)filter_j - y2) + fabs((float)filter_i - x2))/(float)filter_size) *
////									((float)filter_j > y2 ? 1.0f: -1.0f)/ (float)filter_size ;
//									exp( - (fabs((float)filter_j - y2) + fabs((float)filter_i - x2))) *
//									((float)filter_j > y2 ? 1.0f: -1.0f) ;
//
//							}
//						}
					}
					gradinput2_data[batch_i * input2_b_stride + 1 * input2_c_stride + h_i * input2_h_stride + w_i] = bot_diff;

					/***
					  STEP 3: calculate the gradients for input3, i.e. the filter
					 ***/
//					for (c_i = 0 ; c_i < channel; c_i ++ ){
//						float gradoutput_value = gradoutput_data[off + c_i * input1_c_stride + h_i * input1_h_stride + w_i];
//
//						for (filter_j = iy2_T; filter_j < iy2_B ; filter_j ++ ){
//							int _filter_j = min(max(0, filter_j),  h - 1);
//							for(filter_i = ix2_L; filter_i< ix2_R ; filter_i++){
//								int _filter_i = min(max(0,filter_i), w - 1);
//
//								gradinput3_data[ batch_i * input3_b_stride + ((filter_j - iy2_T) * filter_size + (filter_i - ix2_L)) * input3_c_stride + h_i * input3_h_stride + w_i] +=
//									gradoutput_value *
//									input1_data[off + c_i * input1_c_stride + _filter_j * input1_h_stride + _filter_i ] *
////									exp( -(fabs((float)filter_j - y2)+ fabs((float)filter_i - x2))/(float) filter_size) ;
//									exp( -(fabs((float)filter_j - y2)+ fabs((float)filter_i - x2))) ;
//
//							}
//						}
//					}
				}
			}
		}
	}

	error = 0;
	return error;

}


int FlowProjectionLayer_cpu_forward(
		THFloatTensor * input1,
        THFloatTensor * count,
		THFloatTensor * output,
		int fillhole
		)
{
//printf("fddfs\n");
	int error = -1;

	int channel = input1->size[1]; //THTensor_size(state, input1, 1);
	if(channel!=2) return error;
	//printf("1\n");
	int batch = input1->size[0];//THCudaTensor_size(state, input1,0);

	int h = input1->size[2];//THCudaTensor_size(state,input1,2);
	int w = input1->size[3];//THCudaTensor_size(state,input1,3);
	if(count->size[2] != h) return error;// to add some checkpoint
	//printf("2\n");
	if(count->size[3] != w) return error;//printf("3\n");

	float * input1_data = THFloatTensor_data(input1);
    float * count_data = THFloatTensor_data(count);
	float * output_data = THFloatTensor_data(output);

	int input1_b_stride = input1->stride[0];
	int input1_c_stride = input1->stride[1];
	int input1_h_stride = input1->stride[2];
	int input1_w_stride = input1->stride[3];

	int count_b_stride = count->stride[0];
//	int count_c_stride = count->stride[1];
	int count_h_stride = count->stride[2];
	int count_w_stride = count->stride[3];
	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;//printf("4\n");
	if(input1_b_stride != output->stride[0]) return error;//printf("5\n");
	if(input1_c_stride != output->stride[1]) return error;//printf("6\n");
	if(count_w_stride !=1) return error;//printf("7\n");

	    //printf("cpu: input1 %f, \t  output %f \n",input1_data[0],output_data[0]);
//	float fillvalue = 0.0f;//Default to be zero for the out of range position.
	int batch_i;int h_i;int w_i;

	for( batch_i = 0 ; batch_i < batch; batch_i++){
		int off = batch_i * input1_b_stride;
		for( h_i = 0; h_i < h ; h_i ++){
			for( w_i = 0; w_i < w; w_i ++){
                float fx = input1_data[ off + 0 * input1_c_stride + h_i * input1_h_stride + w_i];
                float fy = input1_data[ off + 1 * input1_c_stride + h_i * input1_h_stride + w_i];

                float x2 = (float)( w_i ) + fx;
                float y2 = (float)( h_i ) + fy;

                if (x2 >= 0.0f && y2 >= 0.0f && x2 <= (float)(w-1) && y2 <= (float) (h-1)){
                    int ix2_L = (int)(x2);
					int iy2_T = (int)(y2);
					int ix2_R = min(ix2_L+1, w - 1);
					int iy2_B = min(iy2_T+1, h - 1);

                    output_data[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_L ] += - fx;
                    output_data[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_R ] += - fx;
                    output_data[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_L ] += - fx;
                    output_data[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_R ] += - fx;

                    output_data[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_L ] += - fy;
                    output_data[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_R ] += - fy;
                    output_data[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_L ] += - fy;
                    output_data[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_R ] += - fy;

                    count_data[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_L] += 1.0f;
                    count_data[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_R] += 1.0f;
                    count_data[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L] += 1.0f;
                    count_data[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_R] += 1.0f;
                }
			}
        }
    }
	for( batch_i = 0 ; batch_i < batch; batch_i++){
		int off = batch_i * input1_b_stride;
		for( h_i = 0; h_i < h ; h_i ++){
			for( w_i = 0; w_i < w; w_i ++){
                float temp = count_data[batch_i * count_b_stride + 0 + h_i * count_h_stride + w_i] ;
                if(temp > 0.0f){
                    output_data[off + 0 * input1_c_stride + h_i * input1_h_stride + w_i ] /= temp;
                    output_data[off + 1 * input1_c_stride + h_i * input1_h_stride + w_i ] /= temp;
                }
			}
        }
    }

    //fill hole only in test mode
    if(fillhole){
        //not implemented
        printf("Not implemented but implemented in the GPU/CUDA version\n");

    }
	error = 0;
	return error;

}

int FlowProjectionLayer_cpu_backward(
		THFloatTensor * input1,
        THFloatTensor * count,
		THFloatTensor * gradoutput,
		THFloatTensor * gradinput1
		)
{

	int error = -1;

	int channel = input1->size[1]; //THTensor_size(state, input1, 1);
	if(channel!=2) return error;
	int batch = input1->size[0];//THCudaTensor_size(state, input1,0);

	int h = input1->size[2];//THCudaTensor_size(state,input1,2);
	int w = input1->size[3];//THCudaTensor_size(state,input1,3);
	if(count->size[2] != h) return error;// to add some checkpoint
	if(count->size[3] != w) return error;

	float * input1_data = THFloatTensor_data(input1);
    float * count_data = THFloatTensor_data(count);
	float * gradoutput_data = THFloatTensor_data(gradoutput);
	float * gradinput1_data = THFloatTensor_data(gradinput1);

	int input1_b_stride = input1->stride[0];
	int input1_c_stride = input1->stride[1];
	int input1_h_stride = input1->stride[2];
	int input1_w_stride = input1->stride[3];

	int count_b_stride = count->stride[0];
//	int count_c_stride = count->stride[1];
	int count_h_stride = count->stride[2];
	int count_w_stride = count->stride[3];
	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	if(count_w_stride !=1) return error;

	//    printf("cpu: input1 %f, \t  output %f \n",input1_data[0],output_data[0]);
//	float fillvalue = 0.0f;//Default to be zero for the out of range position.
	int batch_i;int h_i;int w_i;

	for( batch_i = 0 ; batch_i < batch; batch_i++){
		int off = batch_i * input1_b_stride;
		for( h_i = 0; h_i < h ; h_i ++){
			for( w_i = 0; w_i < w; w_i ++){
                float fx = input1_data[ off + 0 * input1_c_stride + h_i * input1_h_stride + w_i];
                float fy = input1_data[ off + 1 * input1_c_stride + h_i * input1_h_stride + w_i];

                float x2 = (float)( w_i ) + fx;
                float y2 = (float)( h_i ) + fy;

                if (x2 >= 0.0f && y2 >= 0.0f && x2 <= (float)(w-1) && y2 <= (float) (h-1)){
                    int ix2_L = (int)(x2);
					int iy2_T = (int)(y2);
					int ix2_R = min(ix2_L+1, w - 1);
					int iy2_B = min(iy2_T+1, h - 1);

                    int iu_offset = off + 0 * input1_c_stride + h_i * input1_h_stride + w_i;
                    gradinput1_data[iu_offset] += - gradoutput_data[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_L]/
                                                    count_data[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_L] ;
                    gradinput1_data[iu_offset] += -gradoutput_data[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_R]/
                                                    count_data[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_R]    ;
                    gradinput1_data[iu_offset] += -  gradoutput_data[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_L]/
                                                    count_data[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L]   ;
                    gradinput1_data[iu_offset] += -  gradoutput_data[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_R]/
                                                    count_data[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_R]  ;

                    int iv_offset = off + 1 * input1_c_stride + h_i * input1_h_stride + w_i;
                    gradinput1_data[iv_offset] += -   gradoutput_data[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_L]/
                                                       count_data[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_L]  ;
                    gradinput1_data[iv_offset] += - gradoutput_data[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_R]/
                                                    count_data[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_R]  ;
                    gradinput1_data[iv_offset] += - gradoutput_data[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_L]/
                                                     count_data[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L]   ;
                    gradinput1_data[iv_offset] += - gradoutput_data[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_R]/
                                                      count_data[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_R];

                }
			}
        }
    }

	error = 0;
	return error;

}


int DepthFlowProjectionLayer_cpu_forward(
		THFloatTensor * input1,
		THFloatTensor * input2,
        THFloatTensor * count,
		THFloatTensor * output,
		int fillhole
		)
{
//printf("fddfs\n");
	int error = -1;

	int channel = input1->size[1]; //THTensor_size(state, input1, 1);
	if(channel!=2) return error;
	//printf("1\n");
	int batch = input1->size[0];//THCudaTensor_size(state, input1,0);

	int h = input1->size[2];//THCudaTensor_size(state,input1,2);
	int w = input1->size[3];//THCudaTensor_size(state,input1,3);
	if(input2->size[1] !=1 )  return error;
    if(count->size[2] != h) return error;// to add some checkpoint
	//printf("2\n");
	if(count->size[3] != w) return error;//printf("3\n");

	float * input1_data = THFloatTensor_data(input1);
	float * input2_data = THFloatTensor_data(input2);
    float * count_data = THFloatTensor_data(count);
	float * output_data = THFloatTensor_data(output);

	int input1_b_stride = input1->stride[0];
	int input1_c_stride = input1->stride[1];
	int input1_h_stride = input1->stride[2];
	int input1_w_stride = input1->stride[3];

	int input2_b_stride = input2->stride[0];
//	int input2_c_stride = input2->stride[1];
	int input2_h_stride = input2->stride[2];
	int input2_w_stride = input2->stride[3];

	int count_b_stride = count->stride[0];
//	int count_c_stride = count->stride[1];
	int count_h_stride = count->stride[2];
	int count_w_stride = count->stride[3];

	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;//printf("4\n");
	if(input2_w_stride !=1) return error;//printf("4\n");	if(input1_b_stride != output->stride[0]) return error;//printf("5\n");
	if(input1_c_stride != output->stride[1]) return error;//printf("6\n");
	if(count_w_stride !=1) return error;//printf("7\n");

	    //printf("cpu: input1 %f, \t  output %f \n",input1_data[0],output_data[0]);
//	float fillvalue = 0.0f;//Default to be zero for the out of range position.
	int batch_i;int h_i;int w_i;

	for( batch_i = 0 ; batch_i < batch; batch_i++){
		int off = batch_i * input1_b_stride;
		for( h_i = 0; h_i < h ; h_i ++){
			for( w_i = 0; w_i < w; w_i ++){
                float fx = input1_data[ off + 0 * input1_c_stride + h_i * input1_h_stride + w_i];
                float fy = input1_data[ off + 1 * input1_c_stride + h_i * input1_h_stride + w_i];

                float x2 = (float)( w_i ) + fx;
                float y2 = (float)( h_i ) + fy;

                if (x2 >= 0.0f && y2 >= 0.0f && x2 <= (float)(w-1) && y2 <= (float) (h-1)){
                    int ix2_L = (int)(x2);
					int iy2_T = (int)(y2);
					int ix2_R = min(ix2_L+1, w - 1);
					int iy2_B = min(iy2_T+1, h - 1);

					float temp = input2_data[batch_i * input2_b_stride + 0 + h_i * input2_h_stride + w_i];
//                    temp = temp;

                    output_data[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_L ] += - temp * fx;
                    output_data[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_R ] += - temp * fx;
                    output_data[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_L ] += - temp * fx;
                    output_data[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_R ] += - temp * fx;

                    output_data[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_L ] += - temp * fy;
                    output_data[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_R ] += - temp * fy;
                    output_data[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_L ] += - temp * fy;
                    output_data[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_R ] += - temp * fy;

                    count_data[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_L] += temp * 1.0f;
                    count_data[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_R] += temp * 1.0f;
                    count_data[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L] += temp * 1.0f;
                    count_data[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_R] += temp * 1.0f;
                }
			}
        }
    }
	for( batch_i = 0 ; batch_i < batch; batch_i++){
		int off = batch_i * input1_b_stride;
		for( h_i = 0; h_i < h ; h_i ++){
			for( w_i = 0; w_i < w; w_i ++){
                float temp = count_data[batch_i * count_b_stride + 0 + h_i * count_h_stride + w_i] ;
                if(temp > 0.0f){
                    output_data[off + 0 * input1_c_stride + h_i * input1_h_stride + w_i ] /= temp;
                    output_data[off + 1 * input1_c_stride + h_i * input1_h_stride + w_i ] /= temp;
                }
			}
        }
    }

    //fill hole only in test mode
    if(fillhole){
        //not implemented
        printf("Not implemented but implemented in the GPU/CUDA version\n");

    }
	error = 0;
	return error;

}

int DepthFlowProjectionLayer_cpu_backward(
		THFloatTensor * input1,
		THFloatTensor * input2,
        THFloatTensor * count,
        THFloatTensor * output,
        THFloatTensor * gradoutput,
		THFloatTensor * gradinput1,
		THFloatTensor * gradinput2
		)
{

	int error = -1;

	int channel = input1->size[1]; //THTensor_size(state, input1, 1);
	if(channel!=2) return error;
	int batch = input1->size[0];//THCudaTensor_size(state, input1,0);

	int h = input1->size[2];//THCudaTensor_size(state,input1,2);
	int w = input1->size[3];//THCudaTensor_size(state,input1,3);
	if(input2->size[1] !=1 )  return error;
	if(count->size[2] != h) return error;// to add some checkpoint
	if(count->size[3] != w) return error;

	float * input1_data = THFloatTensor_data(input1);
 	float * input2_data = THFloatTensor_data(input2);
    float * count_data = THFloatTensor_data(count);
    float * output_data = THFloatTensor_data(output);
    float * gradoutput_data = THFloatTensor_data(gradoutput);
	float * gradinput1_data = THFloatTensor_data(gradinput1);
	float * gradinput2_data = THFloatTensor_data(gradinput2);

	int input1_b_stride = input1->stride[0];
	int input1_c_stride = input1->stride[1];
	int input1_h_stride = input1->stride[2];
	int input1_w_stride = input1->stride[3];

	int input2_b_stride = input2->stride[0];
//	int input2_c_stride = input2->stride[1];
	int input2_h_stride = input2->stride[2];
	int input2_w_stride = input2->stride[3];

	int count_b_stride = count->stride[0];
//	int count_c_stride = count->stride[1];
	int count_h_stride = count->stride[2];
	int count_w_stride = count->stride[3];
	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;//printf("4\n");	if(input1_b_stride != output->stride[0]) return error;//printf("5\n");
    if(count_w_stride !=1) return error;

//    printf("cpu: input1 %f, \t  output %f \n",input1_data[0],output_data[0]);
//	float fillvalue = 0.0f;//Default to be zero for the out of range position.
	int batch_i;int h_i;int w_i;

	for( batch_i = 0 ; batch_i < batch; batch_i++){
		int off = batch_i * input1_b_stride;
		for( h_i = 0; h_i < h ; h_i ++){
			for( w_i = 0; w_i < w; w_i ++){
                float fx = input1_data[ off + 0 * input1_c_stride + h_i * input1_h_stride + w_i];
                float fy = input1_data[ off + 1 * input1_c_stride + h_i * input1_h_stride + w_i];

                float x2 = (float)( w_i ) + fx;
                float y2 = (float)( h_i ) + fy;

                if (x2 >= 0.0f && y2 >= 0.0f && x2 <= (float)(w-1) && y2 <= (float) (h-1)){
                    int ix2_L = (int)(x2);
					int iy2_T = (int)(y2);
					int ix2_R = min(ix2_L+1, w - 1);
					int iy2_B = min(iy2_T+1, h - 1);

					float temp = input2_data[batch_i * input2_b_stride + 0 + h_i * input2_h_stride + w_i];

                    int iu_offset = off + 0 * input1_c_stride + h_i * input1_h_stride + w_i;
                    gradinput1_data[iu_offset] += - gradoutput_data[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_L] * temp/
                                                    count_data[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_L] ;
                    gradinput1_data[iu_offset] += -gradoutput_data[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_R]* temp/
                                                    count_data[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_R]    ;
                    gradinput1_data[iu_offset] += -  gradoutput_data[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_L]* temp/
                                                    count_data[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L]   ;
                    gradinput1_data[iu_offset] += -  gradoutput_data[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_R]* temp/
                                                    count_data[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_R]  ;

                    int iv_offset = off + 1 * input1_c_stride + h_i * input1_h_stride + w_i;
                    gradinput1_data[iv_offset] += -   gradoutput_data[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_L]* temp/
                                                       count_data[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_L]  ;
                    gradinput1_data[iv_offset] += - gradoutput_data[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_R]* temp/
                                                    count_data[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_R];
                    gradinput1_data[iv_offset] += - gradoutput_data[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_L]* temp/
                                                     count_data[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L]   ;
                    gradinput1_data[iv_offset] += - gradoutput_data[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_R]* temp/
                                                      count_data[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_R];

                    //
                    int weight_offset = batch_i * input2_b_stride + 0 + h_i * input2_h_stride + w_i;
                    gradinput2_data[weight_offset] += - gradoutput_data[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_L] /
                                                        count_data[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_L] *
                                                        (fx - output_data[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_L ] );
                    gradinput2_data[weight_offset] += -gradoutput_data[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_R] /
                                                        count_data[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_R] *
                                                        (fx - output_data[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_R ] );
                    gradinput2_data[weight_offset] += -gradoutput_data[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_L] /
                                                        count_data[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L] *
                                                        (fx - output_data[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_L ] );
                    gradinput2_data[weight_offset] += -gradoutput_data[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_R] /
                                                        count_data[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_R] *
                                                        (fx - output_data[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_R ] );

                    gradinput2_data[weight_offset] += - gradoutput_data[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_L] /
                                                        count_data[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_L] *
                                                        (fy - output_data[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_L ] );
                    gradinput2_data[weight_offset] += -gradoutput_data[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_R] /
                                                        count_data[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_R] *
                                                        (fy - output_data[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_R ] );
                    gradinput2_data[weight_offset] += -gradoutput_data[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_L] /
                                                        count_data[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L] *
                                                        (fy - output_data[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_L ] );
                    gradinput2_data[weight_offset] += -gradoutput_data[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_R] /
                                                        count_data[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_R] *
                                                        (fy - output_data[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_R ] );
                }
			}
        }
    }

	error = 0;
	return error;

}
int WeightedFlowProjectionLayer_cpu_forward(
		THFloatTensor * input1,
		THFloatTensor * input2,
		THFloatTensor * input3,
        THFloatTensor * count,
        THFloatTensor * weight,
		THFloatTensor * output,
		int fillhole,
		float threshhold
		)
{
//printf("fddfs\n");
	int error = -1;

	int channel = input1->size[1]; //THTensor_size(state, input1, 1);
	if(channel!=2) return error;
	//printf("1\n");
	if(input2->size[1] !=3 )  return error;
	if(input3->size[1] !=3 )  return error;
	int batch = input1->size[0];//THCudaTensor_size(state, input1,0);

	int h = input1->size[2];//THCudaTensor_size(state,input1,2);
	int w = input1->size[3];//THCudaTensor_size(state,input1,3);
	if(count->size[2] != h) return error;// to add some checkpoint
	//printf("2\n");
	if(count->size[3] != w) return error;//printf("3\n");

	float * input1_data = THFloatTensor_data(input1);
	float * input2_data = THFloatTensor_data(input2);
	float * input3_data = THFloatTensor_data(input3);
    float * count_data = THFloatTensor_data(count);
    float * weight_data = THFloatTensor_data(weight);
	float * output_data = THFloatTensor_data(output);

	int input1_b_stride = input1->stride[0];
	int input1_c_stride = input1->stride[1];
	int input1_h_stride = input1->stride[2];
	int input1_w_stride = input1->stride[3];

	int input2_b_stride = input2->stride[0];
	int input2_c_stride = input2->stride[1];
	int input2_h_stride = input2->stride[2];
	int input2_w_stride = input2->stride[3];

	int input3_b_stride = input3->stride[0];
	int input3_c_stride = input3->stride[1];
	int input3_h_stride = input3->stride[2];
	int input3_w_stride = input3->stride[3];

	int count_b_stride = count->stride[0];
//	int count_c_stride = count->stride[1];
	int count_h_stride = count->stride[2];
	int count_w_stride = count->stride[3];

    int weight_b_stride = weight->stride[0];
//	int weight_c_stride = weight->stride[1];
	int weight_h_stride = weight->stride[2];
	int weight_w_stride = weight->stride[3];
	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;//printf("4\n");
	if(input1_b_stride != output->stride[0]) return error;//printf("5\n");
	if(input1_c_stride != output->stride[1]) return error;//printf("6\n");
	if(count_w_stride !=1) return error;//printf("7\n");
	if(weight_w_stride !=1) return error;//printf("7\n");

    //printf("cpu: input1 %f, \t  output %f \n",input1_data[0],output_data[0]);
//	float fillvalue = 0.0f;//Default to be zero for the out of range position.
	int batch_i;int h_i;int w_i;

	for( batch_i = 0 ; batch_i < batch; batch_i++){
		int off = batch_i * input1_b_stride;
		for( h_i = 0; h_i < h ; h_i ++){
			for( w_i = 0; w_i < w; w_i ++){
                float fx = input1_data[ off + 0 * input1_c_stride + h_i * input1_h_stride + w_i];
                float fy = input1_data[ off + 1 * input1_c_stride + h_i * input1_h_stride + w_i];

                float x2 = (float)( w_i ) + fx;
                float y2 = (float)( h_i ) + fy;



                if (x2 >= 0.0f && y2 >= 0.0f && x2 <= (float)(w-1) && y2 <= (float) (h-1)){

                    int x3 = (int)(max(min((float) (w_i ) + 2.0f * fx, (float) (w) - 1.0f), 0.0f));//for calculating the brightness constancy between input2 and input3
                    int y3 = (int)(max(min((float) (h_i ) + 2.0f * fy, (float) (h) - 1.0f), 0.0f));
                    float weight_i = 0.0f;//data1[3],data2[3];
                    int channel_i;
                    for(channel_i = 0; channel_i < 3; channel_i ++){
                        float data1 = input2_data[batch_i * input2_b_stride + channel_i* input2_c_stride +
                                                    h_i * input2_h_stride + w_i * input2_w_stride];
                        float data2 = input3_data[batch_i * input3_b_stride + channel_i *input3_c_stride +
                                            y3 * input3_h_stride + x3 * input3_w_stride];
                        weight_i += fabs(data1 - data2)/3.0f;
                    }
                    weight_i += 1e-8f; //add a small constant for better verification

                    if(weight_i <= threshhold){// only use the low brightness error flow vectors
                    int ix2_L = (int)(x2);
					int iy2_T = (int)(y2);
					int ix2_R = min(ix2_L+1, w - 1);
					int iy2_B = min(iy2_T+1, h - 1);

                    output_data[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_L ] += - fx;
                    output_data[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_R ] += - fx;
                    output_data[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_L ] += - fx;
                    output_data[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_R ] += - fx;

                    output_data[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_L ] += - fy;
                    output_data[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_R ] += - fy;
                    output_data[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_L ] += - fy;
                    output_data[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_R ] += - fy;

                    count_data[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_L] += 1.0f;
                    count_data[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_R] += 1.0f;
                    count_data[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L] += 1.0f;
                    count_data[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_R] += 1.0f;

                    weight_data[batch_i * weight_b_stride + 0 + iy2_T * weight_h_stride + ix2_L ] += weight_i;
                    weight_data[batch_i * weight_b_stride + 0 + iy2_T * weight_h_stride + ix2_R ] += weight_i;
                    weight_data[batch_i * weight_b_stride + 0 + iy2_B * weight_h_stride + ix2_L ] += weight_i;
                    weight_data[batch_i * weight_b_stride + 0 + iy2_B * weight_h_stride + ix2_R ] += weight_i;
                }}
			}
        }
    }
	for( batch_i = 0 ; batch_i < batch; batch_i++){
		int off = batch_i * input1_b_stride;
		for( h_i = 0; h_i < h ; h_i ++){
			for( w_i = 0; w_i < w; w_i ++){
                float temp = count_data[batch_i * count_b_stride + 0 + h_i * count_h_stride + w_i] ;
                if(temp > 0.0f){
                    output_data[off + 0 * input1_c_stride + h_i * input1_h_stride + w_i ] /= temp;
                    output_data[off + 1 * input1_c_stride + h_i * input1_h_stride + w_i ] /= temp;

                    weight_data[batch_i * weight_b_stride + 0 + h_i * weight_h_stride + w_i ] /= temp;
                }
			}
        }
    }

    //fill hole only in test mode
    if(fillhole){
        //not implemented
        printf("Not implemented but implemented in GPU/CUdA version\n");

    }
	error = 0;
	return error;

}

int WeightedFlowProjectionLayer_cpu_backward(
		THFloatTensor * input1,
        THFloatTensor * input2,
		THFloatTensor * input3,
        THFloatTensor * count,
        THFloatTensor * weight,
		THFloatTensor * gradoutput,
		THFloatTensor * gradinput1,
		float threshhold
		)
{

	int error = -1;

	int channel = input1->size[1]; //THTensor_size(state, input1, 1);
	if(channel!=2) return error;
    if(input2->size[1] !=3 )  return error;
	if(input3->size[1] !=3 )  return error;
	int batch = input1->size[0];//THCudaTensor_size(state, input1,0);

	int h = input1->size[2];//THCudaTensor_size(state,input1,2);
	int w = input1->size[3];//THCudaTensor_size(state,input1,3);
	if(count->size[2] != h) return error;// to add some checkpoint
	if(count->size[3] != w) return error;

	float * input1_data = THFloatTensor_data(input1);
    float * input2_data = THFloatTensor_data(input2);
	float * input3_data = THFloatTensor_data(input3);
    float * count_data = THFloatTensor_data(count);
    float * weight_data = THFloatTensor_data(weight);
	float * gradoutput_data = THFloatTensor_data(gradoutput);
	float * gradinput1_data = THFloatTensor_data(gradinput1);

	int input1_b_stride = input1->stride[0];
	int input1_c_stride = input1->stride[1];
	int input1_h_stride = input1->stride[2];
	int input1_w_stride = input1->stride[3];

	int input2_b_stride = input2->stride[0];
	int input2_c_stride = input2->stride[1];
	int input2_h_stride = input2->stride[2];
	int input2_w_stride = input2->stride[3];

	int input3_b_stride = input3->stride[0];
	int input3_c_stride = input3->stride[1];
	int input3_h_stride = input3->stride[2];
	int input3_w_stride = input3->stride[3];

	int count_b_stride = count->stride[0];
//	int count_c_stride = count->stride[1];
	int count_h_stride = count->stride[2];
	int count_w_stride = count->stride[3];

//    int weight_b_stride = weight->stride[0];
//	int weight_c_stride = weight->stride[1];
//	int weight_h_stride = weight->stride[2];
//	int weight_w_stride = weight->stride[3];
	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	if(count_w_stride !=1) return error;
//    if(weight_w_stride  != 1 ) return error;


	//    printf("cpu: input1  %f, \t  output %f \n",input1_data[0],output_data[0]);
//	float fillvalue = 0.0f;//Default to be zero for the out of range position.
	int batch_i;int h_i;int w_i;

	for( batch_i = 0 ; batch_i < batch; batch_i++){
		int off = batch_i * input1_b_stride;
		for( h_i = 0; h_i < h ; h_i ++){
			for( w_i = 0; w_i < w; w_i ++){
                float fx = input1_data[ off + 0 * input1_c_stride + h_i * input1_h_stride + w_i];
                float fy = input1_data[ off + 1 * input1_c_stride + h_i * input1_h_stride + w_i];

                float x2 = (float)( w_i ) + fx;
                float y2 = (float)( h_i ) + fy;


                if (x2 >= 0.0f && y2 >= 0.0f && x2 <= (float)(w-1) && y2 <= (float) (h-1)){

                    int x3 = (int)(max(min((float) (w_i ) + 2.0f * fx, (float) (w) - 1.0f), 0.0f));//for calculating the brightness constancy between input2 and input3
                    int y3 = (int)(max(min((float) (h_i ) + 2.0f * fy, (float) (h) - 1.0f), 0.0f));
                    float weight_i = 0.0f;//data1[3],data2[3];
                    int channel_i;
                    for(channel_i = 0; channel_i < 3; channel_i ++){
                        float data1 = input2_data[batch_i * input2_b_stride + channel_i* input2_c_stride +
                                                    h_i * input2_h_stride + w_i * input2_w_stride];
                        float data2 = input3_data[batch_i * input3_b_stride + channel_i *input3_c_stride +
                                            y3 * input3_h_stride + x3 * input3_w_stride];
                        weight_i += fabs(data1 - data2)/3.0f;
                    }
                    weight_i += 1e-8f; //add a small constant for better verification

                    if(weight_i <= threshhold){// only use the low brightness error flow vectors
                    int ix2_L = (int)(x2);
					int iy2_T = (int)(y2);
					int ix2_R = min(ix2_L+1, w - 1);
					int iy2_B = min(iy2_T+1, h - 1);

                    int iu_offset = off + 0 * input1_c_stride + h_i * input1_h_stride + w_i;
                    gradinput1_data[iu_offset] += - gradoutput_data[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_L]/
                                                    count_data[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_L] ;
                    gradinput1_data[iu_offset] += -gradoutput_data[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_R]/
                                                    count_data[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_R]    ;
                    gradinput1_data[iu_offset] += -  gradoutput_data[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_L]/
                                                    count_data[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L]   ;
                    gradinput1_data[iu_offset] += -  gradoutput_data[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_R]/
                                                    count_data[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_R]  ;

                    int iv_offset = off + 1 * input1_c_stride + h_i * input1_h_stride + w_i;
                    gradinput1_data[iv_offset] += -   gradoutput_data[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_L]/
                                                       count_data[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_L]  ;
                    gradinput1_data[iv_offset] += - gradoutput_data[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_R]/
                                                    count_data[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_R]  ;
                    gradinput1_data[iv_offset] += - gradoutput_data[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_L]/
                                                     count_data[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L]   ;
                    gradinput1_data[iv_offset] += - gradoutput_data[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_R]/
                                                      count_data[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_R];

                }}
			}
        }
    }

	error = 0;
	return error;

}
//int FlowFillholelayer_cpu_forward(
//		THFloatTensor * input1,
//		THFloatTensor * output
//		)
//{
////printf("fddfs\n");
//	int error = -1;
//
//	int channel = input1->size[1]; //THTensor_size(state, input1, 1);
//	if(channel!=2) return error;
//	//printf("1\n");
//	int batch = input1->size[0];//THCudaTensor_size(state, input1,0);
//
//	int h = input1->size[2];//THCudaTensor_size(state,input1,2);
//	int w = input1->size[3];//THCudaTensor_size(state,input1,3);
////	if(count->size[2] != h) return error;// to add some checkpoint
//	//printf("2\n");
////	if(count->size[3] != w) return error;//printf("3\n");
//
//	float * input1_data = THFloatTensor_data(input1);
////    float * count_data = THFloatTensor_data(count);
//	float * output_data = THFloatTensor_data(output);
//
//	int input1_b_stride = input1->stride[0];
//	int input1_c_stride = input1->stride[1];
//	int input1_h_stride = input1->stride[2];
//	int input1_w_stride = input1->stride[3];
//
////	int count_b_stride = count->stride[0];
////	int count_c_stride = count->stride[1];
////	int count_h_stride = count->stride[2];
////	int count_w_stride = count->stride[3];
//	//TODO: do we need to assert the w_stride to be 1
//	if(input1_w_stride !=1) return error;//printf("4\n");
//	if(input1_b_stride != output->stride[0]) return error;//printf("5\n");
//	if(input1_c_stride != output->stride[1]) return error;//printf("6\n");
////	if(count_w_stride !=1) return error;//printf("7\n");
//
//	    //printf("cpu: input1 %f, \t  output %f \n",input1_data[0],output_data[0]);
////	float fillvalue = 0.0f;//Default to be zero for the out of range position.
//	int batch_i;int h_i;int w_i;
//
//	for( batch_i = 0 ; batch_i < batch; batch_i++){
//		int off = batch_i * input1_b_stride;
//		for( h_i = 0; h_i < h ; h_i ++){
//			for( w_i = 0; w_i < w; w_i ++){
//                float fx = input1_data[ off + 0 * input1_c_stride + h_i * input1_h_stride + w_i];
//                float fy = input1_data[ off + 1 * input1_c_stride + h_i * input1_h_stride + w_i];
//
//                float x2 = (float)( w_i ) + fx;
//                float y2 = (float)( h_i ) + fy;
//
//                if (x2 >= 0.0f && y2 >= 0.0f && x2 <= (float)(w-1) && y2 <= (float) (h-1)){
//                    int ix2_L = (int)(x2);
//					int iy2_T = (int)(y2);
//					int ix2_R = min(ix2_L+1, w - 1);
//					int iy2_B = min(iy2_T+1, h - 1);
//
//                    output_data[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_L ] += - fx;
//                    output_data[off + 0 * input1_c_stride + iy2_T * input1_h_stride + ix2_R ] += - fx;
//                    output_data[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_L ] += - fx;
//                    output_data[off + 0 * input1_c_stride + iy2_B * input1_h_stride + ix2_R ] += - fx;
//
//                    output_data[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_L ] += - fy;
//                    output_data[off + 1 * input1_c_stride + iy2_T * input1_h_stride + ix2_R ] += - fy;
//                    output_data[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_L ] += - fy;
//                    output_data[off + 1 * input1_c_stride + iy2_B * input1_h_stride + ix2_R ] += - fy;
//
//                    count_data[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_L] += 1.0f;
//                    count_data[batch_i * count_b_stride + 0 + iy2_T * count_h_stride + ix2_R] += 1.0f;
//                    count_data[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_L] += 1.0f;
//                    count_data[batch_i * count_b_stride + 0 + iy2_B * count_h_stride + ix2_R] += 1.0f;
//                }
//			}
//        }
//    }
//	for( batch_i = 0 ; batch_i < batch; batch_i++){
//		int off = batch_i * input1_b_stride;
//		for( h_i = 0; h_i < h ; h_i ++){
//			for( w_i = 0; w_i < w; w_i ++){
//                float temp = count_data[batch_i * count_b_stride + 0 + h_i * count_h_stride + w_i] ;
//                if(temp > 0.0f){
//                    output_data[off + 0 * input1_c_stride + h_i * input1_h_stride + w_i ] /= temp;
//                    output_data[off + 1 * input1_c_stride + h_i * input1_h_stride + w_i ] /= temp;
//                }
//			}
//        }
//    }
//	error = 0;
//	return error;
//
//}

int WeightLayer_cpu_forward(
		THFloatTensor * input1, 		THFloatTensor * input2, 		
		THFloatTensor * input3,
//		THFloatTensor * flow1_grad,
		THFloatTensor * output, 
		float  lambda_e, float	lambda_v, float Nw
		)
{
	
	int error = -1;
	if(Nw != 3.0f) {printf("Nw must be odd number 3\n"); return error; }


	int channel = input1->size[1]; //THTensor_size(state, input1, 1);
	if(channel!=3) return error;
	int batch = input1->size[0];//THCudaTensor_size(state, input1,0);
	if(input2->size[0] != batch) return error;

	int h = input1->size[2];//THCudaTensor_size(state,input1,2);
	int w = input1->size[3];//THCudaTensor_size(state,input1,3);
	if(input2->size[2] != h) return error;// to add some checkpoint
	if(input2->size[3] != w) return error;

	//int  = input3->size[1];
	//int filter_size = (int)sqrt((float)filter_size2);
	//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));
	if(input3->size[1] != 2) {printf("the flow channel should be 2\n"); return error;}
//	if(flow1_grad->size[1] != 1) {printf("the flow1_grad channel should be 1\n"); return error;}
	if(output->size[1] != 1) {printf("the flow weight channel should be 1\n"); return error;}

	float * input1_data = THFloatTensor_data(input1);
	float * input2_data = THFloatTensor_data(input2);
	float * input3_data = THFloatTensor_data(input3);
//	float * flow1_grad_data = THFloatTensor_data(flow1_grad);
	float * output_data = THFloatTensor_data(output);

	int input1_b_stride = input1->stride[0];
	int input1_c_stride = input1->stride[1];
	int input1_h_stride = input1->stride[2];
	int input1_w_stride = input1->stride[3];

	int input2_b_stride = input2->stride[0]; if (input2_b_stride != input1_b_stride){printf("Two images are not same size\n"); return error;}
	int input2_c_stride = input2->stride[1];
	int input2_h_stride = input2->stride[2];
	int input2_w_stride = input2->stride[3];

	int input3_b_stride = input3->stride[0];
	int input3_c_stride = input3->stride[1];
	int input3_h_stride = input3->stride[2];
	int input3_w_stride = input3->stride[3];
	
//	int flow1_grad_b_stride = flow1_grad->stride[0];
//	int flow1_grad_c_stride = flow1_grad->stride[1];
//	int flow1_grad_h_stride = flow1_grad->stride[2];
//	int flow1_grad_w_stride = flow1_grad->stride[3];
	
	
	int output_b_stride = output->stride[0];
	//int output_c_stride = output->stride[1];
	int output_h_stride = output->stride[2];
	int output_w_stride = output->stride[3];	
	
	
	//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);

	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
	if(input3_w_stride !=1) return error;
//	if(flow1_grad_w_stride !=1) return error;



	//    printf("cpu: input1 %f, \t  output %f \n",input1_data[0],output_data[0]);
//	float fillvalue = 0.0f;//Default to be zero for the out of range position.
	int batch_i;int h_i;int w_i;int c_i;
	//int filter_i,filter_j;
	

	for( batch_i = 0 ; batch_i < batch; batch_i++){
		int offset = batch_i * input1_b_stride;
		for( h_i = 0; h_i < h ; h_i ++){
			for( w_i = 0; w_i < w; w_i ++){

				//read the opticalflow
				float fx = input3_data[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i];
				float fy = input3_data[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i];

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
					float beta =  y2  - (int)(y2);
					
					int m;
					int n;
					
					float err_sum = 0.0f;
//					float sv_sum = 0.0f ;

//        			bool print =  false;
//					if(batch_i == 3 && h_i == 10 && w_i == 30)  print  = true;
					// Nw must be 3, so that -1,0,1 is the range
					for(m = -1; m <= 1; m ++){
						int patch1_m = min(max(0, m + h_i), h-1);
						for(n = -1; n <= 1; n ++){
							int patch1_n = min(max(0, n + w_i), w-1);
							
							int patch2_mT = min(max(0, m + iy2_T), h-1);
							int patch2_nL = min(max(0, n + ix2_L), w-1);
							int patch2_mB = min(max(0, m + iy2_B), h-1);
							int patch2_nR = min(max(0, n + ix2_R), w-1);
							
							for (c_i = 0; c_i < channel; c_i ++){

								float taget_data =
								    (1-alpha)*(1-beta)*input2_data[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nL] +
                                        alpha*(1-beta)*input2_data[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nR] +
                                        (1-alpha)*beta*input2_data[offset + c_i * input2_c_stride + patch2_mB  * input2_h_stride + patch2_nL] +
                                            alpha*beta*input2_data[offset + c_i * input2_c_stride + patch2_mB * input2_h_stride + patch2_nR];
														
								err_sum += fabs(input1_data[offset + c_i * input1_c_stride + patch1_m * input1_h_stride + patch1_n]
											   - taget_data);																
								//if(print){
								//printf("input1 ==> %f, target ==> %f, err_sum ==> %f\n ",
								//   input1_data[offset + c_i * input1_c_stride + patch1_m * input1_h_stride + patch1_n] ,
								//    taget_data,
								//    err_sum);
								//}
							}
//							sv_sum += flow1_grad_data[batch_i * flow1_grad_b_stride + 0 + patch1_m * flow1_grad_h_stride + patch1_n];
						}
					}
					err_sum /= (channel * Nw * Nw);
//					sv_sum /= (Nw *Nw);
					output_data[batch_i * output_b_stride + 0 + h_i * output_h_stride + w_i * output_w_stride]
					    = (1-err_sum/lambda_e)*(1-err_sum/lambda_e);
//						= expf( - err_sum/lambda_e);
//						 - sv_sum/lambda_v) ;
//					if(print){

//					printf("err_sum ==> %f, sv_sum ==> %f, output ==> %f \n",
//					    err_sum,sv_sum,
//					    output_data[batch_i * output_b_stride + 0 + h_i * output_h_stride + w_i * output_w_stride]);
//					}
					
				}
				else {
					output_data[batch_i * output_b_stride + 0 + h_i * output_h_stride + w_i * output_w_stride]
						= 1e-4f;
				}
			}
		}
	}


	error = 0;
	return error;
}
int WeightLayer_cpu_backward(
		THFloatTensor * input1, 		THFloatTensor * input2, 		THFloatTensor * input3,
//		  THFloatTensor * flow1_grad,
        THFloatTensor * output,
		THFloatTensor * gradoutput, 
		THFloatTensor * gradinput1,		THFloatTensor * gradinput2, 	THFloatTensor * gradinput3,
//		THFloatTensor *  gradflow1_grad,
		float  lambda_e, float	lambda_v, float Nw
		)
{
	
	int error = -1;
	if(Nw != 3.0f) {printf("Nw must be odd number 3\n"); return error;}

	int channel = input1->size[1]; //THTensor_size(state, input1, 1);
	if(channel!=3) return error;
	int batch = input1->size[0];//THCudaTensor_size(state, input1,0);
	if(input2->size[0] != batch) return error;

	int h = input1->size[2];//THCudaTensor_size(state,input1,2);
	int w = input1->size[3];//THCudaTensor_size(state,input1,3);
	if(input2->size[2] != h) return error;// to add some checkpoint
	if(input2->size[3] != w) return error;

	//int  = input3->size[1];
	//int filter_size = (int)sqrt((float)filter_size2);
	//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));
	if(input3->size[1] != 2) {printf("the flow channel should be 2\n"); return error;}
//	if(flow1_grad->size[1] != 1) {printf("the flow1_grad channel should be 1\n"); return error;}
	if(output->size[1] != 1) {printf("the flow weight channel should be 1\n"); return error;}

	float * input1_data = THFloatTensor_data(input1);
	float * input2_data = THFloatTensor_data(input2);
	float * input3_data = THFloatTensor_data(input3);
//	float * flow1_grad_data = THFloatTensor_data(flow1_grad);
	float * output_data = THFloatTensor_data(output);
	float * gradoutput_data = THFloatTensor_data(gradoutput);
	float * gradinput1_data = THFloatTensor_data(gradinput1);
	float * gradinput2_data = THFloatTensor_data(gradinput2);
	float * gradinput3_data = THFloatTensor_data(gradinput3);
//	float * gradflow1_grad_data = THFloatTensor_data(gradflow1_grad);
	

	int input1_b_stride = input1->stride[0];
	int input1_c_stride = input1->stride[1];
	int input1_h_stride = input1->stride[2];
	int input1_w_stride = input1->stride[3];

	int input2_b_stride = input2->stride[0];if (input2_b_stride != input1_b_stride){printf("Two images are not same size\n"); return error;}
	int input2_c_stride = input2->stride[1];
	int input2_h_stride = input2->stride[2];
	int input2_w_stride = input2->stride[3];

	int input3_b_stride = input3->stride[0];
	int input3_c_stride = input3->stride[1];
	int input3_h_stride = input3->stride[2];
	int input3_w_stride = input3->stride[3];
	
//	int flow1_grad_b_stride = flow1_grad->stride[0];
	//int flow1_grad_c_stride = flow1_grad->stride[1];
//	int flow1_grad_h_stride = flow1_grad->stride[2];
//	int flow1_grad_w_stride = flow1_grad->stride[3];
	
	
	int output_b_stride = output->stride[0];
	//int output_c_stride = output->stride[1];
	int output_h_stride = output->stride[2];
	int output_w_stride = output->stride[3];	
	
	
	//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);

	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
	if(input3_w_stride !=1) return error;
//	if(flow1_grad_w_stride !=1) return error;



	//    printf("cpu: input1 %f, \t  output %f \n",input1_data[0],output_data[0]);
//	float fillvalue = 0.0f;//Default to be zero for the out of range position.
	int batch_i;int h_i;int w_i;int c_i;
	//int filter_i,filter_j;
	

	for( batch_i = 0 ; batch_i < batch; batch_i++){
		int offset = batch_i * input1_b_stride;
		for( h_i = 0; h_i < h ; h_i ++){
			for( w_i = 0; w_i < w; w_i ++){
				//read the opticalflow
				float fx = input3_data[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i];
				float fy = input3_data[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i];

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
					float beta =  y2  - (int)(y2);
				 
					float gradoutput_data_value = gradoutput_data[batch_i * output_b_stride + 0 + h_i * output_h_stride + w_i * output_w_stride];
					
					float grad_err_sum = -  gradoutput_data_value / (lambda_e * channel * Nw *Nw)
									* 2 *sqrtf(output_data [batch_i * output_b_stride + 0 + h_i * output_h_stride + w_i * output_w_stride]) ;
//					float grad_sv_sum =  - gradoutput_data_value / (lambda_v * Nw * Nw)
//									* output_data [batch_i * output_b_stride + 0 + h_i * output_h_stride + w_i * output_w_stride] ;
					
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
							
							for (c_i = 0; c_i < 3; c_i ++){								
								
								float taget_data =
	    						    (1-alpha)*(1-beta)*input2_data[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nL] +
                                        alpha*(1-beta)*input2_data[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nR] +
                                        (1-alpha)*beta*input2_data[offset + c_i * input2_c_stride + patch2_mB  * input2_h_stride + patch2_nL] +
                                            alpha*beta*input2_data[offset + c_i * input2_c_stride + patch2_mB * input2_h_stride + patch2_nR] ;
							
								//err_sum += abs(input1_data[offset + c_i * input1_c_stride + patch1_m * input1_h_stride + patch1_n] 
									//		   - taget_data);	

								float i_data = input1_data[offset + c_i * input1_c_stride + patch1_m * input1_h_stride + patch1_n] ;
								
								
								//input1 gradients
								gradinput1_data[offset + c_i * input1_c_stride + patch1_m * input1_h_stride + patch1_n] += 
													( i_data > taget_data) ?  grad_err_sum : - grad_err_sum;
								 
								 //input2 gradients
								gradinput2_data[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nL] += 
								(1-alpha)*(1-beta)*(( i_data> taget_data) ? - grad_err_sum :  grad_err_sum);	
								 
								gradinput2_data[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nR] += 
								alpha*(1-beta)*( ( i_data> taget_data) ?  - grad_err_sum :  grad_err_sum);	
								 
								gradinput2_data[offset + c_i * input2_c_stride + patch2_mB  * input2_h_stride + patch2_nL]  += 
								(1-alpha)*beta*( ( i_data > taget_data) ?   - grad_err_sum :  grad_err_sum);	
								
								gradinput2_data[offset + c_i * input2_c_stride + patch2_mB * input2_h_stride + patch2_nR]   += 
								alpha*beta*( ( i_data > taget_data) ?   - grad_err_sum :  grad_err_sum);									
								
								//input3 gradients
								float gamma  = 1.0f - beta; //iy2_B - y2;
								float temp = 0.0f;
								temp += gamma * (input2_data[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nR]-
										input2_data[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nL]);
								temp += (1-gamma) *( input2_data[offset + c_i * input2_c_stride + patch2_mB * input2_h_stride + patch2_nR]  -
													input2_data[offset + c_i * input2_c_stride + patch2_mB  * input2_h_stride + patch2_nL]);
								
								temp = temp * ( ( i_data > taget_data) ? - grad_err_sum : grad_err_sum);
								gradinput3_data[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i] +=
									temp;
									
								gamma = 1.0f - alpha; //ix2_R -x2;
								temp = 0.0f;
								temp += gamma * ( input2_data[offset + c_i * input2_c_stride + patch2_mB  * input2_h_stride + patch2_nL] - 
													input2_data[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nL]);
								temp += gamma *(input2_data[offset + c_i * input2_c_stride + patch2_mB * input2_h_stride + patch2_nR]  -
												input2_data[offset + c_i * input2_c_stride + patch2_mT * input2_h_stride + patch2_nR] );
								temp = temp * ( ( i_data > taget_data) ? - grad_err_sum : grad_err_sum);
								gradinput3_data[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i] +=
									temp;														
							}
							//flow1_grad's gradients
							//sv_sum += flow1_grad_data[batch_i * flow1_grad_b_stride + 0 + patch1_m * flow1_grad_h_stride + patch1_n];
//							gradflow1_grad_data[ batch_i * flow1_grad_b_stride + 0 + patch1_m * flow1_grad_h_stride + patch1_n] +=
//								grad_sv_sum;
						}
					}		
				}
			}
		}
	}

	error = 0;
	return error;
}
		
int PixelValueLayer_cpu_forward(
		THFloatTensor * input1,  		THFloatTensor * input3, 	THFloatTensor * flow_weights, 		THFloatTensor * output,
		float sigma_d,    float tao_r ,  float Prowindow
		)
{
	
	int error = -1;
	if(Prowindow != 2.0f) {printf("Prowindow must be odd number 2\n"); return error;}

	int channel = input1->size[1]; //THTensor_size(state, input1, 1);
	if(channel!=3) return error;
	int batch = input1->size[0];//THCudaTensor_size(state, input1,0);
	//if(input2->size[0] != batch) return error;

	int h = input1->size[2];//THCudaTensor_size(state,input1,2);
	int w = input1->size[3];//THCudaTensor_size(state,input1,3);
	//if(input2->size[2] != h) return error;// to add some checkpoint
	//if(input2->size[3] != w) return error;

	//int  = input3->size[1];
	//int filter_size = (int)sqrt((float)filter_size2);
	//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));
	if(input3->size[1] != 2) {printf("the flow channel should be 2\n"); return error;}
	if(flow_weights->size[1] != 1) {printf("the flow_weights channel should be 1\n"); return error;}
	if(output->size[1] != 3) {printf("the image channel should be 3\n"); return error;}

	float * input1_data = THFloatTensor_data(input1);
	//float * input2_data = THFloatTensor_data(input2);
	float * input3_data = THFloatTensor_data(input3);
	float * flow_weights_data = THFloatTensor_data(flow_weights);
	float * output_data = THFloatTensor_data(output);

	int input1_b_stride = input1->stride[0];
	int input1_c_stride = input1->stride[1];
	int input1_h_stride = input1->stride[2];
	int input1_w_stride = input1->stride[3];

	//int input2_b_stride = input2->stride[0];
	//int input2_c_stride = input2->stride[1];
	//int input2_h_stride = input2->stride[2];
	//int input2_w_stride = input2->stride[3];

	int input3_b_stride = input3->stride[0];
	int input3_c_stride = input3->stride[1];
	int input3_h_stride = input3->stride[2];
	int input3_w_stride = input3->stride[3];
	
	int flow_weights_b_stride = flow_weights->stride[0];
	//int flow_weights_c_stride = flow_weights->stride[1];
	int flow_weights_h_stride = flow_weights->stride[2];
	int flow_weights_w_stride = flow_weights->stride[3];	
	
	
	int output_b_stride = output->stride[0];
	int output_c_stride = output->stride[1];
	int output_h_stride = output->stride[2];
	int output_w_stride = output->stride[3];	
	
	
	//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);

	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	//if(input2_w_stride !=1) return error;
	if(input3_w_stride !=1) return error;
	if(flow_weights_w_stride !=1) return error; 
	if(output_b_stride != input1_b_stride) return error;
	if(output_c_stride != input1_c_stride) return error;
	if(output_h_stride != input1_h_stride) return error;
	if(output_w_stride != input1_w_stride) return error;
	



	//    printf("cpu: input1 %f, \t  output %f \n",input1_data[0],output_data[0]);
//	float fillvalue = 0.0f;//Default to be zero for the out of range position.
	int batch_i;int h_i;int w_i;int c_i;
	//int filter_i,filter_j;
	

	for( batch_i = 0 ; batch_i < batch; batch_i++){
		int offset = batch_i * input1_b_stride;
		for( h_i = 0; h_i < h ; h_i ++){
			for( w_i = 0; w_i < w; w_i ++){

				//read the opticalflow
				float fx = input3_data[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i];
				float fy = input3_data[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i];
				
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
							
//							float g_d =  expf( - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2 * sigma_d * sigma_d));
                            float g_d = (1.0f - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
                            g_d = g_d * g_d;
                            float f_w = flow_weights_data[batch_i * flow_weights_b_stride + 0 + h_i * flow_weights_h_stride + w_i]  ;
							for(c_i = 0 ; c_i <channel; c_i ++){
								output_data[offset + c_i * output_c_stride +  patch2_m * output_h_stride + patch2_n] += 
									f_w * g_d * input1_data[offset + c_i * input1_c_stride + h_i * input1_h_stride + w_i];
							}
						}												
					}											
				}
			}
		}
	}	

	error = 0;
	return error;
}
int PixelValueLayer_cpu_backward(
		THFloatTensor * input1,  		THFloatTensor * input3, 	THFloatTensor * flow_weights,  
		THFloatTensor * gradoutput, 
		THFloatTensor * gradinput1,		THFloatTensor * gradinput3, THFloatTensor * gradflow_weights,
		float sigma_d,    float tao_r ,  float Prowindow
		)
{
	
	
	int error = -1;
	if(Prowindow != 2.0f) {printf("Prowindow must be odd number 2\n"); return error;}

	int channel = input1->size[1]; //THTensor_size(state, input1, 1);
	if(channel!=3) return error;
	int batch = input1->size[0];//THCudaTensor_size(state, input1,0);
	//if(input2->size[0] != batch) return error;

	int h = input1->size[2];//THCudaTensor_size(state,input1,2);
	int w = input1->size[3];//THCudaTensor_size(state,input1,3);
	//if(input2->size[2] != h) return error;// to add some checkpoint
	//if(input2->size[3] != w) return error;

	//int  = input3->size[1];
	//int filter_size = (int)sqrt((float)filter_size2);
	//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));
	if(input3->size[1] != 2) {printf("the flow channel should be 2\n"); return error;}
	if(flow_weights->size[1] != 1) {printf("the flow_weights channel should be 1\n"); return error;}
	//if(output->size[1] != 3) printf("the flow weight channel should be 1\n"); return error;

	float * input1_data = THFloatTensor_data(input1);
	//float * input2_data = THFloatTensor_data(input2);
	float * input3_data = THFloatTensor_data(input3);
	float * flow_weights_data = THFloatTensor_data(flow_weights);
	//float * output_data = THFloatTensor_data(output);
	float * gradoutput_data = THFloatTensor_data(gradoutput);
	float * gradinput1_data = THFloatTensor_data(gradinput1);
	float * gradinput3_data = THFloatTensor_data(gradinput3);
	float * gradflow_weights_data = THFloatTensor_data(gradflow_weights);


	int input1_b_stride = input1->stride[0];
	int input1_c_stride = input1->stride[1];
	int input1_h_stride = input1->stride[2];
	int input1_w_stride = input1->stride[3];

	//int input2_b_stride = input2->stride[0];
	//int input2_c_stride = input2->stride[1];
	//int input2_h_stride = input2->stride[2];
	//int input2_w_stride = input2->stride[3];

	int input3_b_stride = input3->stride[0];
	int input3_c_stride = input3->stride[1];
	int input3_h_stride = input3->stride[2];
	int input3_w_stride = input3->stride[3];
	
	int flow_weights_b_stride = flow_weights->stride[0];
	//int flow_weights_c_stride = flow_weights->stride[1];
	int flow_weights_h_stride = flow_weights->stride[2];
	int flow_weights_w_stride = flow_weights->stride[3];	
	
	
	//int output_b_stride = output->stride[0];
	//int output_c_stride = output->stride[1];
	//int output_h_stride = output->stride[2];
	//int output_w_stride = output->stride[3];	
	
	
	//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);

	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	//if(input2_w_stride !=1) return error;
	if(input3_w_stride !=1) return error;
	if(flow_weights_w_stride !=1) return error; 



	//    printf("cpu: input1 %f, \t  output %f \n",input1_data[0],output_data[0]);
//	float fillvalue = 0.0f;//Default to be zero for the out of range position.
	int batch_i;int h_i;int w_i;int c_i;
	//int filter_i,filter_j;
	

	for( batch_i = 0 ; batch_i < batch; batch_i++){
		int offset = batch_i * input1_b_stride;
		for( h_i = 0; h_i < h ; h_i ++){
			for( w_i = 0; w_i < w; w_i ++){

				//read the opticalflow
				float fx = input3_data[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i];
				float fy = input3_data[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i];
				
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
							
//                            float g_d =  expf( - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
                            float g_d = (1.0f - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
//                            g_d = g_d * g_d;
                            float f_w = flow_weights_data[batch_i * flow_weights_b_stride + 0 + h_i * flow_weights_h_stride + w_i]  ;
							
							for(c_i = 0 ; c_i < channel; c_i ++){
								float gradoutput_data_value = gradoutput_data[offset + c_i * input1_c_stride +  patch2_m * input1_h_stride + patch2_n];

								//input1 gradients
								gradinput1_data[offset + c_i * input1_c_stride + h_i * input1_h_stride + w_i] += 
									gradoutput_data_value * f_w * g_d * g_d;
								
								// flow_weights_data gradients
								gradflow_weights_data[batch_i * flow_weights_b_stride + 0+ h_i * flow_weights_h_stride + w_i] +=
									gradoutput_data_value * g_d * g_d * input1_data[offset + c_i * input1_c_stride + h_i * input1_h_stride + w_i];
									
								//flow gradients
								gradinput3_data[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i] +=
										- gradoutput_data_value * f_w * input1_data[offset + c_i * input1_c_stride + h_i * input1_h_stride + w_i] *
											g_d * (n - alpha) / (  sigma_d * sigma_d) * 2.0f;
								gradinput3_data[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i] +=
										- gradoutput_data_value * f_w * input1_data[offset + c_i * input1_c_stride + h_i * input1_h_stride + w_i] *
											g_d * (m - beta) / (  sigma_d * sigma_d) * 2.0f;
							}
						}												
					}											
				}
			}
		}
	}
	

	error = 0;
	return error;
}
int PixelWeightLayer_cpu_forward(
		THFloatTensor * input3, 	THFloatTensor * flow_weights, 		THFloatTensor * output,
		float sigma_d,    float tao_r ,  float Prowindow
		)
{
	
	int error = -1;
	if(Prowindow != 2.0f) {printf("Prowindow must be odd number 2\n"); return error;}

	//int channel = input3->size[1]; //THTensor_size(state, input1, 1);
	//if(channel!=3) return error;
	int batch = input3->size[0];//THCudaTensor_size(state, input1,0);
	//if(input2->size[0] != batch) return error;

	int h = input3->size[2];//THCudaTensor_size(state,input1,2);
	int w = input3->size[3];//THCudaTensor_size(state,input1,3);
	//if(input2->size[2] != h) return error;// to add some checkpoint
	//if(input2->size[3] != w) return error;

	//int  = input3->size[1];
	//int filter_size = (int)sqrt((float)filter_size2);
	//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));
	if(input3->size[1] != 2) {printf("the flow channel should be 2\n"); return error;}
	if(flow_weights->size[1] != 1) {printf("the flow_weights channel should be 1\n"); return error;}
	if(output->size[1] != 1) {printf("the flow weight channel should be 1\n"); return error;}

	//float * input1_data = THFloatTensor_data(input1);
	//float * input2_data = THFloatTensor_data(input2);
	float * input3_data = THFloatTensor_data(input3);
	float * flow_weights_data = THFloatTensor_data(flow_weights);
	float * output_data = THFloatTensor_data(output);

	//int input1_b_stride = input1->stride[0];
	//int input1_c_stride = input1->stride[1];
	//int input1_h_stride = input1->stride[2];
	//int input1_w_stride = input1->stride[3];

	//int input2_b_stride = input2->stride[0];
	//int input2_c_stride = input2->stride[1];
	//int input2_h_stride = input2->stride[2];
	//int input2_w_stride = input2->stride[3];

	int input3_b_stride = input3->stride[0];
	int input3_c_stride = input3->stride[1];
	int input3_h_stride = input3->stride[2];
	int input3_w_stride = input3->stride[3];
	
	int flow_weights_b_stride = flow_weights->stride[0];
	//int flow_weights_c_stride = flow_weights->stride[1];
	int flow_weights_h_stride = flow_weights->stride[2];
	int flow_weights_w_stride = flow_weights->stride[3];	
	
	
	int output_b_stride = output->stride[0];
	//int output_c_stride = output->stride[1];
	int output_h_stride = output->stride[2];
	int output_w_stride = output->stride[3];	
	
	
	//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);

	//TODO: do we need to assert the w_stride to be 1
	//if(input1_w_stride !=1) return error;
	if(input3_w_stride !=1) return error;
	if(output_w_stride !=1) return error;
	if(flow_weights_w_stride !=1) return error; 
 

	//    printf("cpu: input1 %f, \t  output %f \n",input1_data[0],output_data[0]);
//	float fillvalue = 0.0f;//Default to be zero for the out of range position.
	int batch_i;int h_i;int w_i;
  	

	for( batch_i = 0 ; batch_i < batch; batch_i++){
		//int off = batch_i * input1_b_stride;
		for( h_i = 0; h_i < h ; h_i ++){
			for( w_i = 0; w_i < w; w_i ++){

				//read the opticalflow
				float fx = input3_data[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i];
				float fy = input3_data[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i];
				
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
							
//							float g_d =  expf( - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
                            float g_d = (1.0f - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
                            g_d = g_d * g_d;
							float f_w = flow_weights_data[batch_i * flow_weights_b_stride + 0 + h_i * flow_weights_h_stride + w_i]  ;
 							output_data[batch_i * output_b_stride + 0 +  patch2_m * output_h_stride + patch2_n] +=  f_w * g_d;
 						}												
					}											
				}
			}
		}
	}	

	error = 0;
	return error;
}
int PixelWeightLayer_cpu_backward(
		THFloatTensor * input3, 	THFloatTensor * flow_weights, 	THFloatTensor * output,
		THFloatTensor * gradoutput, 
		THFloatTensor * gradinput3, THFloatTensor * gradflow_weights,
				float threshhold,
		float sigma_d,    float tao_r ,  float Prowindow
		)
{
	
	int error = -1;
	if(Prowindow != 2.0f) {printf("Prowindow must be odd number 2\n"); return error;}

	//int channel = input1->size[1]; //THTensor_size(state, input1, 1);
	//if(channel!=3) return error;
	int batch = input3->size[0];//THCudaTensor_size(state, input1,0);
	//if(input2->size[0] != batch) return error;

	int h = input3->size[2];//THCudaTensor_size(state,input1,2);
	int w = input3->size[3];//THCudaTensor_size(state,input1,3);
	//if(input2->size[2] != h) return error;// to add some checkpoint
	//if(input2->size[3] != w) return error;

	//int  = input3->size[1];
	//int filter_size = (int)sqrt((float)filter_size2);
	//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));
	if(input3->size[1] != 2) {printf("the flow channel should be 2\n"); return error;}
	if(flow_weights->size[1] != 1) {printf("the flow_weights channel should be 1\n"); return error;}
	//if(output->size[1] != 1) {printf("the flow weight channel should be 1\n"); return error;

	//float * input1_data = THFloatTensor_data(input1);
	//float * input2_data = THFloatTensor_data(input2);
	float * input3_data = THFloatTensor_data(input3);
	float * flow_weights_data = THFloatTensor_data(flow_weights);
	float * output_data = THFloatTensor_data(output);
	float * gradoutput_data = THFloatTensor_data(gradoutput);
	float * gradinput3_data = THFloatTensor_data(gradinput3);
	float * gradflow_weights_data = THFloatTensor_data(gradflow_weights);

	//int input1_b_stride = input1->stride[0];
	//int input1_c_stride = input1->stride[1];
	/////int input1_h_stride = input1->stride[2];
	//int input1_w_stride = input1->stride[3];

	//int input2_b_stride = input2->stride[0];
	//int input2_c_stride = input2->stride[1];
	//int input2_h_stride = input2->stride[2];
	//int input2_w_stride = input2->stride[3];

	int input3_b_stride = input3->stride[0];
	int input3_c_stride = input3->stride[1];
	int input3_h_stride = input3->stride[2];
	int input3_w_stride = input3->stride[3];
	
	int flow_weights_b_stride = flow_weights->stride[0];
	//int flow_weights_c_stride = flow_weights->stride[1];
	int flow_weights_h_stride = flow_weights->stride[2];
	int flow_weights_w_stride = flow_weights->stride[3];	
	
	
	int output_b_stride = gradoutput->stride[0];
	//int output_c_stride = gradoutput->stride[1];
	int output_h_stride = gradoutput->stride[2];
	int output_w_stride = gradoutput->stride[3];	
	
	
	//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);

	//TODO: do we need to assert the w_stride to be 1
	//if(input1_w_stride !=1) return error;
	if(input3_w_stride !=1) return error;
	if(output_w_stride !=1) return error;
	if(flow_weights_w_stride !=1) return error; 



	//    printf("cpu: input1 %f, \t  output %f \n",input1_data[0],output_data[0]);
//	float fillvalue = 0.0f;//Default to be zero for the out of range position.
	int batch_i;int h_i;int w_i;


	for( batch_i = 0 ; batch_i < batch; batch_i++){
		for( h_i = 0; h_i < h ; h_i ++){
			for( w_i = 0; w_i < w; w_i ++){

				//read the opticalflow
				float fx = input3_data[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i];
				float fy = input3_data[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i];
				
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
							
//							float g_d = expf( - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
							float g_d = (1.0f - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
//                            g_d = g_d * g_d;
                            float f_w = flow_weights_data[batch_i * flow_weights_b_stride + 0+ h_i * flow_weights_h_stride + w_i]  ;
							
							//for(c_i = 0 ; c_i < channel; c_i ++){
                            float gradoutput_data_value = gradoutput_data[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n];
                            //if(output_data[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n] < 0)
							 //   printf("Error g_d ==> %f at (%d,0,%d,%d) \n",output_data[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n],batch_i,h_i,w_i );
                            if(output_data[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n] < threshhold)
                             {
                                //printf("pixelweigths cpu backward, under threshhold ==> %f\n",
                                 //output_data[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n]);
                                 continue;//to  skip its gradients
                            }

                            //flow1_weights gradients
                            gradflow_weights_data[batch_i * flow_weights_b_stride + 0+ h_i * flow_weights_h_stride + w_i] +=
                                gradoutput_data_value * g_d * g_d;
									
                            //flow gradients
                            gradinput3_data[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i] +=
                                    - gradoutput_data_value * f_w * g_d * (n - alpha) / ( sigma_d * sigma_d) * 2.0f;
                            gradinput3_data[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i] +=
                                    - gradoutput_data_value * f_w *  g_d *(m - beta)  / (  sigma_d * sigma_d) * 2.0f;
						}												
					}											
				}
			}
		}
	}
	

	error = 0;
	return error;
}		
//int ReliableValueLayer_cpu_forward(
//		THFloatTensor * input3, 	THFloatTensor * flow_weights, 		THFloatTensor * output,
//		float sigma_d,    float tao_r ,  float Prowindow
//		)
//{
//	
//	
//}
//int ReliableValueLayer_cpu_backward(
//		THFloatTensor * input3, 	THFloatTensor * flow_weights, 		THFloatTensor * output,
//		THFloatTensor * gradinput3, THFloatTensor * gradflow_weights,
//		float sigma_d,    float tao_r ,  float Prowindow
//	)
//{
//	
//	
//}
int ReliableWeightLayer_cpu_forward(
		THFloatTensor * input3,  THFloatTensor * output,
		float sigma_d,    float tao_r ,  float Prowindow
		)
{
		int error = -1;
	if(Prowindow != 2.0f) {printf("Prowindow must be odd number 2\n"); return error;}

	//int channel = input3->size[1]; //THTensor_size(state, input1, 1);
	//if(channel!=3) return error;
	int batch = input3->size[0];//THCudaTensor_size(state, input1,0);
	//if(input2->size[0] != batch) return error;

	int h = input3->size[2];//THCudaTensor_size(state,input1,2);
	int w = input3->size[3];//THCudaTensor_size(state,input1,3);
	//if(input2->size[2] != h) return error;// to add some checkpoint
	//if(input2->size[3] != w) return error;

	//int  = input3->size[1];
	//int filter_size = (int)sqrt((float)filter_size2);
	//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));
	if(input3->size[1] != 2) {printf("the flow channel should be 2\n"); return error;}
	//if(flow_weights->size[1] != 1) printf("the flow_weights channel should be 1\n"); return error;
	if(output->size[1] != 1) {printf("the flow weight channel should be 1\n"); return error;}

	//float * input1_data = THFloatTensor_data(input1);
	//float * input2_data = THFloatTensor_data(input2);
	float * input3_data = THFloatTensor_data(input3);
	//float * flow_weights_data = THFloatTensor_data(flow_weights);
	float * output_data = THFloatTensor_data(output);

	//int input1_b_stride = input1->stride[0];
	//int input1_c_stride = input1->stride[1];
	//int input1_h_stride = input1->stride[2];
	//int input1_w_stride = input1->stride[3];

	//int input2_b_stride = input2->stride[0];
	//int input2_c_stride = input2->stride[1];
	//int input2_h_stride = input2->stride[2];
	//int input2_w_stride = input2->stride[3];

	int input3_b_stride = input3->stride[0];
	int input3_c_stride = input3->stride[1];
	int input3_h_stride = input3->stride[2];
	int input3_w_stride = input3->stride[3];
	
	//int flow_weights_b_stride = flow_weights->stride[0];
	//int flow_weights_c_stride = flow_weights->stride[1];
	//int flow_weights_h_stride = flow_weights->stride[2];
	//int flow_weights_w_stride = flow_weights->stride[3];	
	
	
	int output_b_stride = output->stride[0];
	//int output_c_stride = output->stride[1];
	int output_h_stride = output->stride[2];
	int output_w_stride = output->stride[3];	
	
	
	//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);

	//TODO: do we need to assert the w_stride to be 1
	//if(input1_w_stride !=1) return error;
	if(output_w_stride !=1) return error;
	if(input3_w_stride !=1) return error;
	//if(flow_weights_w_stride !=1) return error; 



	//    printf("cpu: input1 %f, \t  output %f \n",input1_data[0],output_data[0]);
//	float fillvalue = 0.0f;//Default to be zero for the out of range position.
	int batch_i;int h_i;int w_i;
	//int c_i;
	//int filter_i,filter_j;
	

	for( batch_i = 0 ; batch_i < batch; batch_i++){
		//int off = batch_i * input1_b_stride;
		for( h_i = 0; h_i < h ; h_i ++){
			for( w_i = 0; w_i < w; w_i ++){

				//read the opticalflow
				float fx = input3_data[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i];
				float fy = input3_data[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i];
				
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
//							float g_d =  expf( - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
                            float g_d = (1.0f - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
                            g_d = g_d * g_d;
							output_data[batch_i * output_b_stride + 0 +  patch2_m * output_h_stride + patch2_n] +=  g_d;
 						}												
					}											
				}else{
					;
					
				}
			}
		}
	}	

	error = 0;
	return error;
} 
int ReliableWeightLayer_cpu_backward(
		THFloatTensor * input3, 	 		THFloatTensor * output,
		THFloatTensor * gradoutput, 
		THFloatTensor * gradinput3,  
        float threshhold,
		float sigma_d,    float tao_r ,  float Prowindow
	)
{
		
	int error = -1;
	if(Prowindow != 2.0f) {printf("Prowindow must be odd number 2\n"); return error;}

	//int channel = input1->size[1]; //THTensor_size(state, input1, 1);
	//if(channel!=3) return error;
	int batch = input3->size[0];//THCudaTensor_size(state, input1,0);
	//if(input2->size[0] != batch) return error;

	int h = input3->size[2];//THCudaTensor_size(state,input1,2);
	int w = input3->size[3];//THCudaTensor_size(state,input1,3);
	//if(input2->size[2] != h) return error;// to add some checkpoint
	//if(input2->size[3] != w) return error;

	//int  = input3->size[1];
	//int filter_size = (int)sqrt((float)filter_size2);
	//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));
	if(input3->size[1] != 2) {printf("the flow channel should be 2\n"); return error;}
	//if(flow_weights->size[1] != 1) printf("the flow_weights channel should be 1\n"); return error;
	//if(output->size[1] != 1) printf("the flow weight channel should be 1\n"); return error;

	//float * input1_data = THFloatTensor_data(input1);
	//float * input2_data = THFloatTensor_data(input2);
	float * input3_data = THFloatTensor_data(input3);
	//float * flow_weights_data = THFloatTensor_data(flow_weights);
	float * output_data = THFloatTensor_data(output);
	float * gradoutput_data = THFloatTensor_data(gradoutput);
	float * gradinput3_data = THFloatTensor_data(gradinput3);
	
	
	//int input1_b_stride = input1->stride[0];
	//int input1_c_stride = input1->stride[1];
	/////int input1_h_stride = input1->stride[2];
	//int input1_w_stride = input1->stride[3];

	//int input2_b_stride = input2->stride[0];
	//int input2_c_stride = input2->stride[1];
	//int input2_h_stride = input2->stride[2];
	//int input2_w_stride = input2->stride[3];

	int input3_b_stride = input3->stride[0];
	int input3_c_stride = input3->stride[1];
	int input3_h_stride = input3->stride[2];
	int input3_w_stride = input3->stride[3];
	
	//int flow_weights_b_stride = flow_weights->stride[0];
	//int flow_weights_c_stride = flow_weights->stride[1];
	///int flow_weights_h_stride = flow_weights->stride[2];
	//int flow_weights_w_stride = flow_weights->stride[3];	
	
	
	int output_b_stride = gradoutput->stride[0];
	//int output_c_stride = gradoutput->stride[1];
	int output_h_stride = gradoutput->stride[2];
	int output_w_stride = gradoutput->stride[3];	
	
	
	//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);

	//TODO: do we need to assert the w_stride to be 1
	//if(input1_w_stride !=1) return error;
	if(output_w_stride !=1) return error;
	if(input3_w_stride !=1) return error;
	//if(flow_weights_w_stride !=1) return error; 



	//    printf("cpu: input1 %f, \t  output %f \n",input1_data[0],output_data[0]);
//	float fillvalue = 0.0f;//Default to be zero for the out of range position.
	int batch_i;int h_i;int w_i;
	//int c_i;
	//int filter_i,filter_j;
	

	for( batch_i = 0 ; batch_i < batch; batch_i++){
 		for( h_i = 0; h_i < h ; h_i ++){
			for( w_i = 0; w_i < w; w_i ++){

				//read the opticalflow
				float fx = input3_data[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i];
				float fy = input3_data[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i];
				
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
							
//							float g_d =  expf( - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
 						    float g_d = (1.0f - ((beta - m) * (beta - m) + (alpha - n)  *( alpha - n ))/(2.0f * sigma_d * sigma_d));
//                            g_d = g_d * g_d;
							float gradoutput_data_value = gradoutput_data[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n];
							//if(output_data[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n] < 0)
							//    printf("Error g_d ==> %f \n",output_data[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n] );
                            if(output_data[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n] < threshhold)
                            {
                                //printf("ReliableWeights cpu backward, under threshhold ==> %f\n",
                                 //output_data[batch_i * output_b_stride + 0  +  patch2_m * output_h_stride + patch2_n]);
                                 continue;//to  skip its gradients
                            }
							//flow gradients
							gradinput3_data[batch_i * input3_b_stride + 0 * input3_c_stride + h_i * input3_h_stride + w_i] +=
									- gradoutput_data_value *  g_d * (n - alpha) / ( sigma_d * sigma_d) * 2.0f;
							gradinput3_data[batch_i * input3_b_stride + 1 * input3_c_stride + h_i * input3_h_stride + w_i] +=
									- gradoutput_data_value *  g_d * (m - beta)  / (  sigma_d * sigma_d) * 2.0f;
 						}												
					}											
				}
			}
		}
	} 

	error = 0;
	return error;
}
