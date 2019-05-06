#include <THC.h>
//#include <THCGeneral.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include "my_lib_kernel.h" //MUST be included

//an important state pointer for the management
extern THCState* state;


int SeparableConvFlowLayer_gpu_forward(
		THCudaTensor * input1,
		THCudaTensor * input2,
		THCudaTensor * input3,
		//THCudaTensor * output,
		THCudaTensor* flow_output

		)
		{
	int error = - 1;
    //int point  =0 ;printf("debug point  %d\n", point++ );

	int channel = THCudaTensor_size(state, input1, 1);
	if(channel!=3) return error;
	int batch = THCudaTensor_size(state, input1,0);
	if(THCudaTensor_size(state, input2, 0) != batch) return error;
	if(THCudaTensor_size(state, input2,1) != THCudaTensor_size(state, input2,1)) return error;
    //printf("debug point  %d\n", point++ );

	int h = THCudaTensor_size(state,input1,2);
	int w = THCudaTensor_size(state,input1,3);
	if(THCudaTensor_size(state,input2,2) != h - THCudaTensor_size(state, input2,1) + 1) return error;// to add some checkpoint
	if(THCudaTensor_size(state,input2,3) != w - THCudaTensor_size(state, input2,1) + 1) return error;
	

	int input1_b_stride = THCudaTensor_stride(state, input1,0);
	int input1_c_stride = THCudaTensor_stride(state, input1,1);
	int input1_h_stride = THCudaTensor_stride(state, input1,2);
	int input1_w_stride = THCudaTensor_stride(state, input1,3);

	int input2_b_stride = THCudaTensor_stride(state, input2,0);
	int input2_c_stride = THCudaTensor_stride(state, input2,1);
	int input2_h_stride = THCudaTensor_stride(state, input2,2);
	int input2_w_stride = THCudaTensor_stride(state, input2,3);

    int input3_b_stride = THCudaTensor_stride(state, input3,0);
	int input3_c_stride = THCudaTensor_stride(state, input3,1);
	int input3_h_stride = THCudaTensor_stride(state, input3,2);
	int input3_w_stride = THCudaTensor_stride(state, input3,3);

    //int output_b_stride = THCudaTensor_stride(state, output,0);
	//int output_c_stride = THCudaTensor_stride(state, output,1);
	//int output_h_stride = THCudaTensor_stride(state, output,2);
	//int output_w_stride = THCudaTensor_stride(state, output,3);
	
    int flow_output_b_stride = THCudaTensor_stride(state, flow_output,0);
	int flow_output_c_stride = THCudaTensor_stride(state, flow_output,1);
	int flow_output_h_stride = THCudaTensor_stride(state, flow_output,2);
	int flow_output_w_stride = THCudaTensor_stride(state, flow_output,3);	
    //printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
    if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
   // if(output_w_stride !=1) return error;
	if(flow_output_w_stride !=1) return error;


	if(input2_b_stride != input3_b_stride) return error;
	if(input2_c_stride != input3_c_stride) return error;
    //printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	int	nElement = THCudaTensor_nElement(state, flow_output);


	error = SeparableConvFlowLayer_gpu_forward_kernel(
			THCState_getCurrentStream(state),
			nElement,w,h,channel,batch,  THCudaTensor_size(state, input2,1),

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
		//	output_b_stride,output_c_stride,output_h_stride,output_w_stride,
			flow_output_b_stride,flow_output_c_stride,flow_output_h_stride,flow_output_w_stride,



			THCudaTensor_data(state,input1),
			THCudaTensor_data(state,input2),
			THCudaTensor_data(state,input3),
			//THCudaTensor_data(state,output ),
			THCudaTensor_data(state,flow_output )
			
			);
	THCudaCheck(cudaGetLastError());
    //printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);

	return error;

		}
int SeparableConvFlowLayer_gpu_backward(
		THCudaTensor * input1,
		THCudaTensor * input2,
		THCudaTensor * input3,
		THCudaTensor * gradflow_output,
		THCudaTensor * gradinput1,
		THCudaTensor * gradinput2,
		THCudaTensor * gradinput3
		)
		{


    int error = - 1;
	int channel = THCudaTensor_size(state, input1, 1);
	if(channel!=3) return error;
	int batch = THCudaTensor_size(state, input1,0);
	if(THCudaTensor_size(state,input2, 0) != batch) return error;
	if(THCudaTensor_size(state, input2,1) != THCudaTensor_size(state, input2,1)) return error;

	int h = THCudaTensor_size(state,input1,2);
	int w = THCudaTensor_size(state,input1,3);
	if(THCudaTensor_size(state,input2,2) != h - THCudaTensor_size(state, input2,1) + 1) return error;// to add some checkpoint
	if(THCudaTensor_size(state,input2,3) != w - THCudaTensor_size(state, input2,1) + 1) return error;


	int input1_b_stride = THCudaTensor_stride(state, input1,0);
	int input1_c_stride = THCudaTensor_stride(state, input1,1);
	int input1_h_stride = THCudaTensor_stride(state, input1,2);
	int input1_w_stride = THCudaTensor_stride(state, input1,3);

	int input2_b_stride = THCudaTensor_stride(state, input2,0);
	int input2_c_stride = THCudaTensor_stride(state, input2,1);
	int input2_h_stride = THCudaTensor_stride(state, input2,2);
	int input2_w_stride = THCudaTensor_stride(state, input2,3);

    int input3_b_stride = THCudaTensor_stride(state, input3,0);
	int input3_c_stride = THCudaTensor_stride(state, input3,1);
	int input3_h_stride = THCudaTensor_stride(state, input3,2);
	int input3_w_stride = THCudaTensor_stride(state, input3,3);

    //int output_b_stride = THCudaTensor_stride(state, gradoutput,0);
	//int output_c_stride = THCudaTensor_stride(state, gradoutput,1);
	//int output_h_stride = THCudaTensor_stride(state, gradoutput,2);
	//int output_w_stride = THCudaTensor_stride(state, gradoutput,3);
	
    int flow_output_b_stride = THCudaTensor_stride(state, gradflow_output,0);
	int flow_output_c_stride = THCudaTensor_stride(state, gradflow_output,1);
	int flow_output_h_stride = THCudaTensor_stride(state, gradflow_output,2);
	int flow_output_w_stride = THCudaTensor_stride(state, gradflow_output,3);		

//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
  //  if(output_w_stride !=1) return error;
	if(flow_output_w_stride !=1) return error;

    if(input1_b_stride != THCudaTensor_stride(state, gradinput1,0)) return error;
	if(input2_b_stride != THCudaTensor_stride(state, gradinput2,0)) return error;
	if(input1_c_stride != THCudaTensor_stride(state, gradinput1,1)) return error;
	if(input2_c_stride != THCudaTensor_stride(state, gradinput2,1)) return error;
	if(input3_c_stride != THCudaTensor_stride(state, gradinput3,1)) return error;

//    printf("GPU backward: %d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);

	int	nElement = THCudaTensor_nElement(state, gradflow_output);

	error  = SeparableConvFlowLayer_gpu_backward_kernel(
			THCState_getCurrentStream(state),
			nElement, //to let the nummous
			w,h,channel,batch,  THCudaTensor_size(state, input2,1),

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
		//	output_b_stride,output_c_stride,output_h_stride,output_w_stride,
			flow_output_b_stride,flow_output_c_stride,flow_output_h_stride,flow_output_w_stride,

			THCudaTensor_data(state,input1),
			THCudaTensor_data(state,input2),
			THCudaTensor_data(state,input3),
			THCudaTensor_data(state,gradflow_output),
			THCudaTensor_data(state,gradinput1),
			THCudaTensor_data(state,gradinput2),
			THCudaTensor_data(state,gradinput3)
			);
	THCudaCheck(cudaGetLastError());

	return error;
}


int SeparableConvLayer_gpu_forward(
		THCudaTensor * input1,
		THCudaTensor * input2,
		THCudaTensor * input3,
		THCudaTensor * output

		)
		{
	int error = - 1;

	int channel = THCudaTensor_size(state, input1, 1);
	if(channel!=3) return error;
	int batch = THCudaTensor_size(state, input1,0);
	if(THCudaTensor_size(state, input2, 0) != batch) return error;
	if(THCudaTensor_size(state, input2,1) != THCudaTensor_size(state, input2,1)) return error;

	int h = THCudaTensor_size(state,input1,2);
	int w = THCudaTensor_size(state,input1,3);
	if(THCudaTensor_size(state,input2,2) != h - THCudaTensor_size(state, input2,1) + 1) return error;// to add some checkpoint
	if(THCudaTensor_size(state,input2,3) != w - THCudaTensor_size(state, input2,1) + 1) return error;


	int input1_b_stride = THCudaTensor_stride(state, input1,0);
	int input1_c_stride = THCudaTensor_stride(state, input1,1);
	int input1_h_stride = THCudaTensor_stride(state, input1,2);
	int input1_w_stride = THCudaTensor_stride(state, input1,3);

	int input2_b_stride = THCudaTensor_stride(state, input2,0);
	int input2_c_stride = THCudaTensor_stride(state, input2,1);
	int input2_h_stride = THCudaTensor_stride(state, input2,2);
	int input2_w_stride = THCudaTensor_stride(state, input2,3);

    int input3_b_stride = THCudaTensor_stride(state, input3,0);
	int input3_c_stride = THCudaTensor_stride(state, input3,1);
	int input3_h_stride = THCudaTensor_stride(state, input3,2);
	int input3_w_stride = THCudaTensor_stride(state, input3,3);

    int output_b_stride = THCudaTensor_stride(state, output,0);
	int output_c_stride = THCudaTensor_stride(state, output,1);
	int output_h_stride = THCudaTensor_stride(state, output,2);
	int output_w_stride = THCudaTensor_stride(state, output,3);
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
    if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
    if(output_w_stride !=1) return error;

	if(input2_b_stride != input3_b_stride) return error;
	if(input2_c_stride != input3_c_stride) return error;


	int	nElement = THCudaTensor_nElement(state, output);


	error = SeparableConvLayer_gpu_forward_kernel(
			THCState_getCurrentStream(state),
			nElement,w,h,channel,batch,  THCudaTensor_size(state, input2,1),

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,



			THCudaTensor_data(state,input1),
			THCudaTensor_data(state,input2),
			THCudaTensor_data(state,input3),
			THCudaTensor_data(state,output ));
	THCudaCheck(cudaGetLastError());

	return error;

		}
int SeparableConvLayer_gpu_backward(
		THCudaTensor * input1,
		THCudaTensor * input2,
		THCudaTensor * input3,
		THCudaTensor * gradoutput,
		THCudaTensor * gradinput1,
		THCudaTensor * gradinput2,
		THCudaTensor * gradinput3
		)
		{


    int error = - 1;
	int channel = THCudaTensor_size(state, input1, 1);
	if(channel!=3) return error;
	int batch = THCudaTensor_size(state, input1,0);
	if(THCudaTensor_size(state,input2, 0) != batch) return error;
	if(THCudaTensor_size(state, input2,1) != THCudaTensor_size(state, input2,1)) return error;

	int h = THCudaTensor_size(state,input1,2);
	int w = THCudaTensor_size(state,input1,3);
	if(THCudaTensor_size(state,input2,2) != h - THCudaTensor_size(state, input2,1) + 1) return error;// to add some checkpoint
	if(THCudaTensor_size(state,input2,3) != w - THCudaTensor_size(state, input2,1) + 1) return error;


	int input1_b_stride = THCudaTensor_stride(state, input1,0);
	int input1_c_stride = THCudaTensor_stride(state, input1,1);
	int input1_h_stride = THCudaTensor_stride(state, input1,2);
	int input1_w_stride = THCudaTensor_stride(state, input1,3);

	int input2_b_stride = THCudaTensor_stride(state, input2,0);
	int input2_c_stride = THCudaTensor_stride(state, input2,1);
	int input2_h_stride = THCudaTensor_stride(state, input2,2);
	int input2_w_stride = THCudaTensor_stride(state, input2,3);

    int input3_b_stride = THCudaTensor_stride(state, input3,0);
	int input3_c_stride = THCudaTensor_stride(state, input3,1);
	int input3_h_stride = THCudaTensor_stride(state, input3,2);
	int input3_w_stride = THCudaTensor_stride(state, input3,3);

    int output_b_stride = THCudaTensor_stride(state, gradoutput,0);
	int output_c_stride = THCudaTensor_stride(state, gradoutput,1);
	int output_h_stride = THCudaTensor_stride(state, gradoutput,2);
	int output_w_stride = THCudaTensor_stride(state, gradoutput,3);

//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
    if(output_w_stride !=1) return error;

    if(input1_b_stride != THCudaTensor_stride(state, gradinput1,0)) return error;
	if(input2_b_stride != THCudaTensor_stride(state, gradinput2,0)) return error;
	if(input1_c_stride != THCudaTensor_stride(state, gradinput1,1)) return error;
	if(input2_c_stride != THCudaTensor_stride(state, gradinput2,1)) return error;
	if(input3_c_stride != THCudaTensor_stride(state, gradinput3,1)) return error;

//    printf("GPU backward: %d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);

	int	nElement = THCudaTensor_nElement(state, gradoutput);

	error  = SeparableConvLayer_gpu_backward_kernel(
			THCState_getCurrentStream(state),
			nElement, //to let the nummous
			w,h,channel,batch,  THCudaTensor_size(state, input2,1),

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,

			THCudaTensor_data(state,input1),
			THCudaTensor_data(state,input2),
			THCudaTensor_data(state,input3),
			THCudaTensor_data(state,gradoutput),
			THCudaTensor_data(state,gradinput1),
			THCudaTensor_data(state,gradinput2),
			THCudaTensor_data(state,gradinput3)
			);
	THCudaCheck(cudaGetLastError());

	return error;
}

int InterpolationLayer_gpu_forward(
		THCudaTensor * input1,
		THCudaTensor * input2,
		THCudaTensor * output
		)
		{
	int error = - 1;

	int channel = THCudaTensor_size(state, input1, 1);
	if(channel!=3) return error;
	int batch = THCudaTensor_size(state, input1,0);
	if(THCudaTensor_size(state,input2, 0) != batch) return error;
	if(THCudaTensor_size(state, input2,1) != 2) return error;

	int h = THCudaTensor_size(state,input1,2);
	int w = THCudaTensor_size(state,input1,3);
	if(THCudaTensor_size(state,input2,2) != h) return error;// to add some checkpoint
	if(THCudaTensor_size(state,input2,3) != w) return error;

//	float * input1_data = THCudaTensor_data(state,input1);
//	float * input2_data = THCudaTensor_data(state,input2);

	int input1_b_stride = THCudaTensor_stride(state, input1,0);
	int input1_c_stride = THCudaTensor_stride(state, input1,1);
	int input1_h_stride = THCudaTensor_stride(state, input1,2);
	int input1_w_stride = THCudaTensor_stride(state, input1,3);

	int input2_b_stride = THCudaTensor_stride(state, input2,0);
	int input2_c_stride = THCudaTensor_stride(state, input2,1);
	int input2_h_stride = THCudaTensor_stride(state, input2,2);
	int input2_w_stride = THCudaTensor_stride(state, input2,3);
	//TODO: do we need to assert the w_stride to be 1
	//if(w_stride !=1) return error;
	if(input1_b_stride != THCudaTensor_stride(state, output,0)) return error;
	if(input1_c_stride != THCudaTensor_stride(state, output,1)) return error;

	int	nElement = THCudaTensor_nElement(state, output);

	error =InterpolationLayer_gpu_forward_kernel(
			THCState_getCurrentStream(state),
			nElement,w,h,channel,batch,

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,

			THCudaTensor_data(state,input1),
			THCudaTensor_data(state,input2),
			THCudaTensor_data(state,output ));
	THCudaCheck(cudaGetLastError());

	return error;

}


int InterpolationLayer_gpu_backward(
		THCudaTensor* input1,
		THCudaTensor* input2,
		THCudaTensor * gradoutput,
		THCudaTensor * gradinput1,
		THCudaTensor * gradinput2
		)
    {
	int error = - 1;
	int channel = THCudaTensor_size(state, input1, 1);
	if(channel!=3) return error;
	int batch = THCudaTensor_size(state, input1,0);
	if(THCudaTensor_size(state,input2, 0) != batch) return error;
	if(THCudaTensor_size(state, input2,1) != 2) return error;

	int h = THCudaTensor_size(state,input1,2);
	int w = THCudaTensor_size(state,input1,3);
	if(THCudaTensor_size(state,input2,2) != h) return error;// to add some checkpoint
	if(THCudaTensor_size(state,input2,3) != w) return error;

//	float * input1_data = THCudaTensor_data(state,input1);
//	float * input2_data = THCudaTensor_data(state,input2);

	int input1_b_stride = THCudaTensor_stride(state, input1,0);
	int input1_c_stride = THCudaTensor_stride(state, input1,1);
	int input1_h_stride = THCudaTensor_stride(state, input1,2);
	int input1_w_stride = THCudaTensor_stride(state, input1,3);

	int input2_b_stride = THCudaTensor_stride(state, input2,0);
	int input2_c_stride = THCudaTensor_stride(state, input2,1);
	int input2_h_stride = THCudaTensor_stride(state, input2,2);
	int input2_w_stride = THCudaTensor_stride(state, input2,3);
	//TODO: do we need to assert the w_stride to be 1
	//if(w_stride !=1) return error;
	if(input1_b_stride != THCudaTensor_stride(state, gradinput1,0)) return error;
	if(input2_b_stride != THCudaTensor_stride(state, gradinput2,0)) return error;
	if(input1_c_stride != THCudaTensor_stride(state, gradinput1,1)) return error;
	if(input2_c_stride != THCudaTensor_stride(state, gradinput2,1)) return error;

//    printf("GPU backward: %d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);

	int	nElement = THCudaTensor_nElement(state, gradoutput);

	error  = InterpolationLayer_gpu_backward_kernel(
			THCState_getCurrentStream(state),
			nElement, //to let the nummous
			w,h,channel,batch,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,

			THCudaTensor_data(state,input1),
			THCudaTensor_data(state,input2),
			THCudaTensor_data(state,gradoutput),
			THCudaTensor_data(state,gradinput1),
			THCudaTensor_data(state,gradinput2)
			);
	THCudaCheck(cudaGetLastError());

	return error;

}

int InterpolationChLayer_gpu_forward(
		THCudaTensor * input1,
		THCudaTensor * input2,
		THCudaTensor * output
		)
		{
	int error = - 1;

	int channel = THCudaTensor_size(state, input1, 1);
//	if(channel!=3) return error;
	int batch = THCudaTensor_size(state, input1,0);
	if(THCudaTensor_size(state,input2, 0) != batch) return error;
	if(THCudaTensor_size(state, input2,1) != 2) return error;

	int h = THCudaTensor_size(state,input1,2);
	int w = THCudaTensor_size(state,input1,3);
	if(THCudaTensor_size(state,input2,2) != h) return error;// to add some checkpoint
	if(THCudaTensor_size(state,input2,3) != w) return error;

//	float * input1_data = THCudaTensor_data(state,input1);
//	float * input2_data = THCudaTensor_data(state,input2);

	int input1_b_stride = THCudaTensor_stride(state, input1,0);
	int input1_c_stride = THCudaTensor_stride(state, input1,1);
	int input1_h_stride = THCudaTensor_stride(state, input1,2);
	int input1_w_stride = THCudaTensor_stride(state, input1,3);

	int input2_b_stride = THCudaTensor_stride(state, input2,0);
	int input2_c_stride = THCudaTensor_stride(state, input2,1);
	int input2_h_stride = THCudaTensor_stride(state, input2,2);
	int input2_w_stride = THCudaTensor_stride(state, input2,3);
	//TODO: do we need to assert the w_stride to be 1
	//if(w_stride !=1) return error;
	if(input1_b_stride != THCudaTensor_stride(state, output,0)) return error;
	if(input1_c_stride != THCudaTensor_stride(state, output,1)) return error;

	int	nElement = THCudaTensor_nElement(state, output);

	error =InterpolationLayer_gpu_forward_kernel(
			THCState_getCurrentStream(state),
			nElement,w,h,channel,batch,

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,

			THCudaTensor_data(state,input1),
			THCudaTensor_data(state,input2),
			THCudaTensor_data(state,output ));
	THCudaCheck(cudaGetLastError());

	return error;

}


int InterpolationChLayer_gpu_backward(
		THCudaTensor* input1,
		THCudaTensor* input2,
		THCudaTensor * gradoutput,
		THCudaTensor * gradinput1,
		THCudaTensor * gradinput2
		)
    {
	int error = - 1;
	int channel = THCudaTensor_size(state, input1, 1);
//	if(channel!=3) return error;
	int batch = THCudaTensor_size(state, input1,0);
	if(THCudaTensor_size(state,input2, 0) != batch) return error;
	if(THCudaTensor_size(state, input2,1) != 2) return error;

	int h = THCudaTensor_size(state,input1,2);
	int w = THCudaTensor_size(state,input1,3);
	if(THCudaTensor_size(state,input2,2) != h) return error;// to add some checkpoint
	if(THCudaTensor_size(state,input2,3) != w) return error;

//	float * input1_data = THCudaTensor_data(state,input1);
//	float * input2_data = THCudaTensor_data(state,input2);

	int input1_b_stride = THCudaTensor_stride(state, input1,0);
	int input1_c_stride = THCudaTensor_stride(state, input1,1);
	int input1_h_stride = THCudaTensor_stride(state, input1,2);
	int input1_w_stride = THCudaTensor_stride(state, input1,3);

	int input2_b_stride = THCudaTensor_stride(state, input2,0);
	int input2_c_stride = THCudaTensor_stride(state, input2,1);
	int input2_h_stride = THCudaTensor_stride(state, input2,2);
	int input2_w_stride = THCudaTensor_stride(state, input2,3);
	//TODO: do we need to assert the w_stride to be 1
	//if(w_stride !=1) return error;
	if(input1_b_stride != THCudaTensor_stride(state, gradinput1,0)) return error;
	if(input2_b_stride != THCudaTensor_stride(state, gradinput2,0)) return error;
	if(input1_c_stride != THCudaTensor_stride(state, gradinput1,1)) return error;
	if(input2_c_stride != THCudaTensor_stride(state, gradinput2,1)) return error;

//    printf("GPU backward: %d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);

	int	nElement = THCudaTensor_nElement(state, gradoutput);

	error  = InterpolationLayer_gpu_backward_kernel(
			THCState_getCurrentStream(state),
			nElement, //to let the nummous
			w,h,channel,batch,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,

			THCudaTensor_data(state,input1),
			THCudaTensor_data(state,input2),
			THCudaTensor_data(state,gradoutput),
			THCudaTensor_data(state,gradinput1),
			THCudaTensor_data(state,gradinput2)
			);
	THCudaCheck(cudaGetLastError());

	return error;

}

int FilterInterpolationLayer_gpu_forward(
		THCudaTensor * input1,
		THCudaTensor * input2,
		THCudaTensor * input3,
		THCudaTensor * output

		)
		{
	int error = - 1;

	int channel = THCudaTensor_size(state, input1, 1);
	//if(channel!=3) return error;
	int batch = THCudaTensor_size(state, input1,0);
	if(THCudaTensor_size(state,input2, 0) != batch) return error;
	if(THCudaTensor_size(state, input2,1) != 2) return error;

	int h = THCudaTensor_size(state,input1,2);
	int w = THCudaTensor_size(state,input1,3);
	if(THCudaTensor_size(state,input2,2) != h) return error;// to add some checkpoint
	if(THCudaTensor_size(state,input2,3) != w) return error;

    int filter_size2 = THCudaTensor_size(state, input3, 1);
    int filter_size = (int) sqrt((float) filter_size2);
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));


	int input1_b_stride = THCudaTensor_stride(state, input1,0);
	int input1_c_stride = THCudaTensor_stride(state, input1,1);
	int input1_h_stride = THCudaTensor_stride(state, input1,2);
	int input1_w_stride = THCudaTensor_stride(state, input1,3);

	int input2_b_stride = THCudaTensor_stride(state, input2,0);
	int input2_c_stride = THCudaTensor_stride(state, input2,1);
	int input2_h_stride = THCudaTensor_stride(state, input2,2);
	int input2_w_stride = THCudaTensor_stride(state, input2,3);

    int input3_b_stride = THCudaTensor_stride(state, input3,0);
	int input3_c_stride = THCudaTensor_stride(state, input3,1);
	int input3_h_stride = THCudaTensor_stride(state, input3,2);
	int input3_w_stride = THCudaTensor_stride(state, input3,3);
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
    if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
	if(input1_b_stride != THCudaTensor_stride(state, output,0)) return error;
	if(input1_c_stride != THCudaTensor_stride(state, output,1)) return error;

	int	nElement = THCudaTensor_nElement(state, output);


	error = FilterInterpolationLayer_gpu_forward_kernel(
			THCState_getCurrentStream(state),
			nElement,w,h,channel,batch, filter_size,

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,


			THCudaTensor_data(state,input1),
			THCudaTensor_data(state,input2),
			THCudaTensor_data(state,input3),
			THCudaTensor_data(state,output ));
	THCudaCheck(cudaGetLastError());

	return error;

		}
int FilterInterpolationLayer_gpu_backward(
		THCudaTensor * input1,
		THCudaTensor * input2,
		THCudaTensor * input3,
		THCudaTensor * gradoutput,
		THCudaTensor * gradinput1,
		THCudaTensor * gradinput2,
		THCudaTensor * gradinput3
		)
		{


    int error = - 1;
	int channel = THCudaTensor_size(state, input1, 1);
	//if(channel!=3) return error;
	int batch = THCudaTensor_size(state, input1,0);
	if(THCudaTensor_size(state,input2, 0) != batch) return error;
	if(THCudaTensor_size(state, input2,1) != 2) return error;

	int h = THCudaTensor_size(state,input1,2);
	int w = THCudaTensor_size(state,input1,3);
	if(THCudaTensor_size(state,input2,2) != h) return error;// to add some checkpoint
	if(THCudaTensor_size(state,input2,3) != w) return error;


    int filter_size2 = THCudaTensor_size(state, input3, 1);
    int filter_size = (int) sqrt((float) filter_size2);
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));

	int input1_b_stride = THCudaTensor_stride(state, input1,0);
	int input1_c_stride = THCudaTensor_stride(state, input1,1);
	int input1_h_stride = THCudaTensor_stride(state, input1,2);
	int input1_w_stride = THCudaTensor_stride(state, input1,3);

	int input2_b_stride = THCudaTensor_stride(state, input2,0);
	int input2_c_stride = THCudaTensor_stride(state, input2,1);
	int input2_h_stride = THCudaTensor_stride(state, input2,2);
	int input2_w_stride = THCudaTensor_stride(state, input2,3);

    int input3_b_stride = THCudaTensor_stride(state, input3,0);
	int input3_c_stride = THCudaTensor_stride(state, input3,1);
	int input3_h_stride = THCudaTensor_stride(state, input3,2);
	int input3_w_stride = THCudaTensor_stride(state, input3,3);
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
    if(input1_b_stride != THCudaTensor_stride(state, gradinput1,0)) return error;
	if(input2_b_stride != THCudaTensor_stride(state, gradinput2,0)) return error;
	if(input1_c_stride != THCudaTensor_stride(state, gradinput1,1)) return error;
	if(input2_c_stride != THCudaTensor_stride(state, gradinput2,1)) return error;
	if(input3_c_stride != THCudaTensor_stride(state, gradinput3,1)) return error;

//    printf("GPU backward: %d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);

	int	nElement = THCudaTensor_nElement(state, gradoutput);

	error  = FilterInterpolationLayer_gpu_backward_kernel(
			THCState_getCurrentStream(state),
			nElement, //to let the nummous
			w,h,channel,batch, filter_size,

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,

			THCudaTensor_data(state,input1),
			THCudaTensor_data(state,input2),
			THCudaTensor_data(state,input3),
			THCudaTensor_data(state,gradoutput),
			THCudaTensor_data(state,gradinput1),
			THCudaTensor_data(state,gradinput2),
			THCudaTensor_data(state,gradinput3)
			);
	THCudaCheck(cudaGetLastError());

	return error;
}


int FlowProjectionLayer_gpu_forward(
		THCudaTensor * input1,
		THCudaTensor * count,
		THCudaTensor * output,
		int fillhole
		)
{

	int error = - 1;

	int channel = THCudaTensor_size(state, input1, 1);
	if(channel!= 2) return error;
	int batch = THCudaTensor_size(state, input1,0);

	int h = THCudaTensor_size(state,input1,2);
	int w = THCudaTensor_size(state,input1,3);

	int input1_b_stride = THCudaTensor_stride(state, input1,0);
	int input1_c_stride = THCudaTensor_stride(state, input1,1);
	int input1_h_stride = THCudaTensor_stride(state, input1,2);
	int input1_w_stride = THCudaTensor_stride(state, input1,3);

	int count_b_stride = THCudaTensor_stride(state, count,0);
	int count_c_stride = THCudaTensor_stride(state, count,1);
	int count_h_stride = THCudaTensor_stride(state, count,2);
	int count_w_stride = THCudaTensor_stride(state, count,3);
	//TODO: do we need to assert the w_stride to be 1
	//if(w_stride !=1) return error;
	if(input1_b_stride != THCudaTensor_stride(state, output,0)) return error;
	if(input1_c_stride != THCudaTensor_stride(state, output,1)) return error;

	int	nElement = THCudaTensor_nElement(state, output);
//    printf("In gpu forward\n");
	error = FlowProjection_gpu_forward_kernel(
			THCState_getCurrentStream(state),
			nElement,w,h,channel,batch,fillhole,

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			count_b_stride,count_c_stride,count_h_stride,count_w_stride,

			THCudaTensor_data(state,input1),
			THCudaTensor_data(state,count),
			THCudaTensor_data(state,output ));
	THCudaCheck(cudaGetLastError());

	return error;

}

int FlowProjectionLayer_gpu_backward(
		THCudaTensor * input1,
        THCudaTensor * count,
		THCudaTensor * gradoutput,
		THCudaTensor * gradinput1
		)
{
	int error = - 1;
	int channel = THCudaTensor_size(state, input1, 1);
	if(channel!=2) return error;
	int batch = THCudaTensor_size(state, input1,0);
	if(THCudaTensor_size(state,count, 0) != batch) return error;
	if(THCudaTensor_size(state, count,1) != 1) return error;

	int h = THCudaTensor_size(state,input1,2);
	int w = THCudaTensor_size(state,input1,3);
	if(THCudaTensor_size(state,count,2) != h) return error;// to add some checkpoint
	if(THCudaTensor_size(state,count,3) != w) return error;

	int input1_b_stride = THCudaTensor_stride(state, input1,0);
	int input1_c_stride = THCudaTensor_stride(state, input1,1);
	int input1_h_stride = THCudaTensor_stride(state, input1,2);
	int input1_w_stride = THCudaTensor_stride(state, input1,3);

	int count_b_stride = THCudaTensor_stride(state, count,0);
	int count_c_stride = THCudaTensor_stride(state, count,1);
	int count_h_stride = THCudaTensor_stride(state, count,2);
	int count_w_stride = THCudaTensor_stride(state, count,3);
	//TODO: do we need to assert the w_stride to be 1
	//if(w_stride !=1) return error;
	if(input1_b_stride != THCudaTensor_stride(state, gradinput1,0)) return error;
	if(input1_c_stride != THCudaTensor_stride(state, gradinput1,1)) return error;

//    printf("GPU backward: %d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);
//    printf("GPU backward: %d,%d,%d,%d\n", count_b_stride,count_c_stride,count_h_stride,count_w_stride);

	int	nElement = THCudaTensor_nElement(state, gradoutput);

	error  = FlowProjection_gpu_backward_kernel(
			THCState_getCurrentStream(state),
			nElement, //to let the nummous
			w,h,channel,batch,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			count_b_stride,count_c_stride,count_h_stride,count_w_stride,

			THCudaTensor_data(state,input1),
			THCudaTensor_data(state,count),
			THCudaTensor_data(state,gradoutput),
			THCudaTensor_data(state,gradinput1)
			);
	THCudaCheck(cudaGetLastError());

	return error;

}

int DepthFlowProjectionLayer_gpu_forward(
		THCudaTensor * input1,
        THCudaTensor * input2,
        THCudaTensor * count,
		THCudaTensor * output,
		int fillhole
		)
{

	int error = - 1;

	int channel = THCudaTensor_size(state, input1, 1);
	if(channel!= 2) return error;
	int batch = THCudaTensor_size(state, input1,0);

	int h = THCudaTensor_size(state,input1,2);
	int w = THCudaTensor_size(state,input1,3);

    if(THCudaTensor_size(state,input2,1) !=1 ) return error;

	int input1_b_stride = THCudaTensor_stride(state, input1,0);
	int input1_c_stride = THCudaTensor_stride(state, input1,1);
	int input1_h_stride = THCudaTensor_stride(state, input1,2);
	int input1_w_stride = THCudaTensor_stride(state, input1,3);

	int input2_b_stride = THCudaTensor_stride(state, input2,0);
	int input2_c_stride = THCudaTensor_stride(state, input2,1);
	int input2_h_stride = THCudaTensor_stride(state, input2,2);
	int input2_w_stride = THCudaTensor_stride(state, input2,3);

	int count_b_stride = THCudaTensor_stride(state, count,0);
	int count_c_stride = THCudaTensor_stride(state, count,1);
	int count_h_stride = THCudaTensor_stride(state, count,2);
	int count_w_stride = THCudaTensor_stride(state, count,3);
	//TODO: do we need to assert the w_stride to be 1
	//if(w_stride !=1) return error;
	if(input1_b_stride != THCudaTensor_stride(state, output,0)) return error;
	if(input1_c_stride != THCudaTensor_stride(state, output,1)) return error;

	int	nElement = THCudaTensor_nElement(state, output);
//    printf("In gpu forward\n");
	error = DepthFlowProjection_gpu_forward_kernel(
			THCState_getCurrentStream(state),
			nElement,w,h,channel,batch,fillhole,

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
            input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
            count_b_stride,count_c_stride,count_h_stride,count_w_stride,

			THCudaTensor_data(state,input1),
			THCudaTensor_data(state,input2),
			THCudaTensor_data(state,count),
			THCudaTensor_data(state,output ));
	THCudaCheck(cudaGetLastError());

	return error;

}

int DepthFlowProjectionLayer_gpu_backward(
		THCudaTensor * input1,
		THCudaTensor * input2,
        THCudaTensor * count,
		THCudaTensor * output,
        THCudaTensor * gradoutput,
		THCudaTensor * gradinput1,
		THCudaTensor * gradinput2
		)
{
	int error = - 1;
	int channel = THCudaTensor_size(state, input1, 1);
	if(channel!=2) return error;
	int batch = THCudaTensor_size(state, input1,0);
	if(THCudaTensor_size(state,count, 0) != batch) return error;
	if(THCudaTensor_size(state, count,1) != 1) return error;

	int h = THCudaTensor_size(state,input1,2);
	int w = THCudaTensor_size(state,input1,3);
    if(THCudaTensor_size(state,input2,1) !=1 ) return error;
    if(THCudaTensor_size(state,count,2) != h) return error;// to add some checkpoint
	if(THCudaTensor_size(state,count,3) != w) return error;

	int input1_b_stride = THCudaTensor_stride(state, input1,0);
	int input1_c_stride = THCudaTensor_stride(state, input1,1);
	int input1_h_stride = THCudaTensor_stride(state, input1,2);
	int input1_w_stride = THCudaTensor_stride(state, input1,3);

	int input2_b_stride = THCudaTensor_stride(state, input2,0);
	int input2_c_stride = THCudaTensor_stride(state, input2,1);
	int input2_h_stride = THCudaTensor_stride(state, input2,2);
	int input2_w_stride = THCudaTensor_stride(state, input2,3);

	int count_b_stride = THCudaTensor_stride(state, count,0);
	int count_c_stride = THCudaTensor_stride(state, count,1);
	int count_h_stride = THCudaTensor_stride(state, count,2);
	int count_w_stride = THCudaTensor_stride(state, count,3);
	//TODO: do we need to assert the w_stride to be 1
	//if(w_stride !=1) return error;
	if(input1_b_stride != THCudaTensor_stride(state, gradinput1,0)) return error;
	if(input1_c_stride != THCudaTensor_stride(state, gradinput1,1)) return error;

//    printf("GPU backward: %d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);
//    printf("GPU backward: %d,%d,%d,%d\n", count_b_stride,count_c_stride,count_h_stride,count_w_stride);

	int	nElement = THCudaTensor_nElement(state, gradoutput);

	error  = DepthFlowProjection_gpu_backward_kernel(
			THCState_getCurrentStream(state),
			nElement, //to let the nummous
			w,h,channel,batch,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
            input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
            count_b_stride,count_c_stride,count_h_stride,count_w_stride,

			THCudaTensor_data(state,input1),
            THCudaTensor_data(state,input2),
            THCudaTensor_data(state,count),
            THCudaTensor_data(state,output),
			THCudaTensor_data(state,gradoutput),
			THCudaTensor_data(state,gradinput1),
			THCudaTensor_data(state,gradinput2)
			);
	THCudaCheck(cudaGetLastError());

	return error;

}


int WeightedFlowProjectionLayer_gpu_forward(
		THCudaTensor * input1,
		THCudaTensor * input2,
		THCudaTensor * input3,
		THCudaTensor * count,
		THCudaTensor * weight,
		THCudaTensor * output,
		int fillhole,
		float threshold
		)
{

	int error = - 1;

	int channel = THCudaTensor_size(state, input1, 1);
	if(channel!= 2) return error;
	if(THCudaTensor_size(state, input2, 1) !=3) return error;
	if(THCudaTensor_size(state, input3, 1) !=3) return error;

	int batch = THCudaTensor_size(state, input1,0);

	int h = THCudaTensor_size(state,input1,2);
	int w = THCudaTensor_size(state,input1,3);

	int input1_b_stride = THCudaTensor_stride(state, input1,0);
	int input1_c_stride = THCudaTensor_stride(state, input1,1);
	int input1_h_stride = THCudaTensor_stride(state, input1,2);
	int input1_w_stride = THCudaTensor_stride(state, input1,3);

	int input2_b_stride = THCudaTensor_stride(state, input2,0);
	int input2_c_stride = THCudaTensor_stride(state, input2,1);
	int input2_h_stride = THCudaTensor_stride(state, input2,2);
	int input2_w_stride = THCudaTensor_stride(state, input2,3);

    int input3_b_stride = THCudaTensor_stride(state, input3,0);
	int input3_c_stride = THCudaTensor_stride(state, input3,1);
	int input3_h_stride = THCudaTensor_stride(state, input3,2);
	int input3_w_stride = THCudaTensor_stride(state, input3,3);

	int count_b_stride = THCudaTensor_stride(state, count,0);
	int count_c_stride = THCudaTensor_stride(state, count,1);
	int count_h_stride = THCudaTensor_stride(state, count,2);
	int count_w_stride = THCudaTensor_stride(state, count,3);

	int weight_b_stride = THCudaTensor_stride(state, weight,0);
	int weight_c_stride = THCudaTensor_stride(state, weight,1);
	int weight_h_stride = THCudaTensor_stride(state, weight,2);
	int weight_w_stride = THCudaTensor_stride(state, weight,3);
	//TODO: do we need to assert the w_stride to be 1
	//if(w_stride !=1) return error;
	if(input1_b_stride != THCudaTensor_stride(state, output,0)) return error;
	if(input1_c_stride != THCudaTensor_stride(state, output,1)) return error;

	int	nElement = THCudaTensor_nElement(state, output);
//    printf("In gpu forward\n");
	error = WeightedFlowProjection_gpu_forward_kernel(
			THCState_getCurrentStream(state),
			nElement,w,h,channel,batch,fillhole, threshold,

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
            input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			count_b_stride,count_c_stride,count_h_stride,count_w_stride,
			weight_b_stride,weight_c_stride,weight_h_stride,weight_w_stride,

			THCudaTensor_data(state,input1),
			THCudaTensor_data(state,input2),
			THCudaTensor_data(state,input3),
			THCudaTensor_data(state,count),
			THCudaTensor_data(state,weight),
			THCudaTensor_data(state,output ));
	THCudaCheck(cudaGetLastError());

	return error;

}

int WeightedFlowProjectionLayer_gpu_backward(
		THCudaTensor * input1,
        THCudaTensor * input2,
		THCudaTensor  * input3,
        THCudaTensor * count,
        THCudaTensor * weight,
		THCudaTensor * gradoutput,
		THCudaTensor * gradinput1,
		float threshhold
		)
{
	int error = - 1;
	int channel = THCudaTensor_size(state, input1, 1);
	if(channel!=2) return error;
	if(THCudaTensor_size(state, input2, 1) !=3) return error;
	if(THCudaTensor_size(state, input3, 1) !=3) return error;
	int batch = THCudaTensor_size(state, input1,0);
	if(THCudaTensor_size(state,count, 0) != batch) return error;
	if(THCudaTensor_size(state, count,1) != 1) return error;

	int h = THCudaTensor_size(state,input1,2);
	int w = THCudaTensor_size(state,input1,3);
	if(THCudaTensor_size(state,count,2) != h) return error;// to add some checkpoint
	if(THCudaTensor_size(state,count,3) != w) return error;

	int input1_b_stride = THCudaTensor_stride(state, input1,0);
	int input1_c_stride = THCudaTensor_stride(state, input1,1);
	int input1_h_stride = THCudaTensor_stride(state, input1,2);
	int input1_w_stride = THCudaTensor_stride(state, input1,3);

	int input2_b_stride = THCudaTensor_stride(state, input2,0);
	int input2_c_stride = THCudaTensor_stride(state, input2,1);
	int input2_h_stride = THCudaTensor_stride(state, input2,2);
	int input2_w_stride = THCudaTensor_stride(state, input2,3);

    int input3_b_stride = THCudaTensor_stride(state, input3,0);
	int input3_c_stride = THCudaTensor_stride(state, input3,1);
	int input3_h_stride = THCudaTensor_stride(state, input3,2);
	int input3_w_stride = THCudaTensor_stride(state, input3,3);

	int count_b_stride = THCudaTensor_stride(state, count,0);
	int count_c_stride = THCudaTensor_stride(state, count,1);
	int count_h_stride = THCudaTensor_stride(state, count,2);
	int count_w_stride = THCudaTensor_stride(state, count,3);

	int weight_b_stride = THCudaTensor_stride(state, weight,0);
	int weight_c_stride = THCudaTensor_stride(state, weight,1);
	int weight_h_stride = THCudaTensor_stride(state, weight,2);
	int weight_w_stride = THCudaTensor_stride(state, weight,3);
	//TODO: do we need to assert the w_stride to be 1
	//if(w_stride !=1) return error;
	if(input1_b_stride != THCudaTensor_stride(state, gradinput1,0)) return error;
	if(input1_c_stride != THCudaTensor_stride(state, gradinput1,1)) return error;

//    printf("GPU backward: %d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);
//    printf("GPU backward: %d,%d,%d,%d\n", count_b_stride,count_c_stride,count_h_stride,count_w_stride);

	int	nElement = THCudaTensor_nElement(state, gradoutput);

	error  = WeightedFlowProjection_gpu_backward_kernel(
			THCState_getCurrentStream(state),
			nElement, //to let the nummous
			w,h,channel,batch, threshhold,
			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
            input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			count_b_stride,count_c_stride,count_h_stride,count_w_stride,
			weight_b_stride,weight_c_stride,weight_h_stride,weight_w_stride,

			THCudaTensor_data(state,input1),
            THCudaTensor_data(state,input2),
			THCudaTensor_data(state,input3),
			THCudaTensor_data(state,count),
            THCudaTensor_data(state,weight),
			THCudaTensor_data(state,gradoutput),
			THCudaTensor_data(state,gradinput1)
			);
	THCudaCheck(cudaGetLastError());

	return error;

}


int WeightLayer_gpu_forward(
		THCudaTensor * input1, 		THCudaTensor * input2, 		THCudaTensor * input3,
//		THCudaTensor * flow1_grad,
    THCudaTensor * output,
		float  lambda_e, float	lambda_v, float Nw
		)
{
	int error = - 1;
	//printf("Entering WeightLayer_gpu_forward\n");
	if(Nw != 3.0f) {printf("Nw must be odd number 3\n"); return error;}

	int channel = THCudaTensor_size(state, input1, 1);
	if(channel!=3) return error;
	int batch = THCudaTensor_size(state, input1,0);
	if(THCudaTensor_size(state,input2, 0) != batch) return error;

	//printf("Entering WeightLayer_gpu_forward\n");
	int h = THCudaTensor_size(state,input1,2);
	int w = THCudaTensor_size(state,input1,3);
	if(THCudaTensor_size(state,input2,2) != h) return error;// to add some checkpoint
	if(THCudaTensor_size(state,input2,3) != w) return error;

	//printf("Entering WeightLayer_gpu_forward\n");
//    int filter_size2 = THCudaTensor_size(state, input3, 1);
  //  int filter_size = (int) sqrt((float) filter_size2);
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));
	if(THCudaTensor_size(state, input3, 1) != 2){ printf("the flow channel should be 2\n"); return error;}
//	if(THCudaTensor_size(state, flow1_grad,1) != 1){ printf("the flow1_grad channel should be 1\n"); return error;}
	if(THCudaTensor_size(state, output,1) != 1) {printf("the flow weight channel should be 1\n"); return error;}


	//printf("Entering WeightLayer_gpu_forward\n");
	int input1_b_stride = THCudaTensor_stride(state, input1,0);
	int input1_c_stride = THCudaTensor_stride(state, input1,1);
	int input1_h_stride = THCudaTensor_stride(state, input1,2);
	int input1_w_stride = THCudaTensor_stride(state, input1,3);

	int input2_b_stride = THCudaTensor_stride(state, input2,0);
	int input2_c_stride = THCudaTensor_stride(state, input2,1);
	int input2_h_stride = THCudaTensor_stride(state, input2,2);
	int input2_w_stride = THCudaTensor_stride(state, input2,3);

    int input3_b_stride = THCudaTensor_stride(state, input3,0);
	int input3_c_stride = THCudaTensor_stride(state, input3,1);
	int input3_h_stride = THCudaTensor_stride(state, input3,2);
	int input3_w_stride = THCudaTensor_stride(state, input3,3);
	
//	int flow1_grad_b_stride = THCudaTensor_stride(state,flow1_grad,0);
//	int flow1_grad_c_stride = THCudaTensor_stride(state,flow1_grad,1);
//	int flow1_grad_h_stride = THCudaTensor_stride(state,flow1_grad,2);
//	int flow1_grad_w_stride = THCudaTensor_stride(state,flow1_grad,3);
	
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);
	int output_b_stride = THCudaTensor_stride(state,output,0);
	int output_c_stride = THCudaTensor_stride(state,output,1);
	int output_h_stride = THCudaTensor_stride(state,output,2);
	int output_w_stride = THCudaTensor_stride(state,output,3);	

	//TODO: do we need to assert the w_stride to be 1
    if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
//    if(flow1_grad_w_stride != 1) return error;
    if(output_w_stride !=1 )return error;
	//printf("Entering WeightLayer_gpu_forward\n");
//	if(input1_b_stride != THCudaTensor_stride(state, output,0)) return error;
//	if(input1_c_stride != THCudaTensor_stride(state, output,1)) return error;

	int	nElement = THCudaTensor_nElement(state, output);


	error = WeightLayer_gpu_forward_kernel(
			THCState_getCurrentStream(state),
			nElement,w,h,channel,batch,  

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
//			flow1_grad_b_stride,flow1_grad_c_stride,flow1_grad_h_stride,flow1_grad_w_stride,
			output_b_stride,  output_c_stride,   output_h_stride,   output_w_stride,

			THCudaTensor_data(state,input1),
			THCudaTensor_data(state,input2),
			THCudaTensor_data(state,input3),
//			THCudaTensor_data(state,flow1_grad),
			THCudaTensor_data(state,output),
			
			lambda_e,  lambda_v,   Nw
			);
	THCudaCheck(cudaGetLastError());

	return error;
	
}
int WeightLayer_gpu_backward(
		THCudaTensor * input1, 		THCudaTensor * input2, 		THCudaTensor * input3,
//		THCudaTensor * flow1_grad,
    THCudaTensor * output,
		THCudaTensor * gradoutput, 
		THCudaTensor * gradinput1,		THCudaTensor * gradinput2, 		THCudaTensor * gradinput3,
//		THCudaTensor *  gradflow1_grad,
		float  lambda_e, float	lambda_v, float Nw
		)
{
		
    int error = - 1;
	if(Nw != 3.0f) {printf("Nw must be odd number 3\n"); return error;}

	int channel = THCudaTensor_size(state, input1, 1);
	if(channel!=3) return error;
	int batch = THCudaTensor_size(state, input1,0);
	if(THCudaTensor_size(state, input2, 0) != batch) return error;
	if(THCudaTensor_size(state, input3,1) != 2) return error;

	int h = THCudaTensor_size(state,input1,2);
	int w = THCudaTensor_size(state,input1,3);
	if(THCudaTensor_size(state,input2,2) != h) return error;// to add some checkpoint
	if(THCudaTensor_size(state,input2,3) != w) return error;


   // int filter_size2 = THCudaTensor_size(state, input3, 1);
    //int filter_size = (int) sqrt((float) filter_size2);
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));
	if(THCudaTensor_size(state, input3, 1) != 2) {printf("the flow channel should be 2\n"); return error;}
//	if(THCudaTensor_size(state, flow1_grad,1) != 1) {printf("the flow1_grad channel should be 1\n"); return error;}
	if(THCudaTensor_size(state, output,1) != 1) {printf("the flow weight channel should be 1\n"); return error;}

	int input1_b_stride = THCudaTensor_stride(state, input1,0);
	int input1_c_stride = THCudaTensor_stride(state, input1,1);
	int input1_h_stride = THCudaTensor_stride(state, input1,2);
	int input1_w_stride = THCudaTensor_stride(state, input1,3);

	int input2_b_stride = THCudaTensor_stride(state, input2,0);
	int input2_c_stride = THCudaTensor_stride(state, input2,1);
	int input2_h_stride = THCudaTensor_stride(state, input2,2);
	int input2_w_stride = THCudaTensor_stride(state, input2,3);

    int input3_b_stride = THCudaTensor_stride(state, input3,0);
	int input3_c_stride = THCudaTensor_stride(state, input3,1);
	int input3_h_stride = THCudaTensor_stride(state, input3,2);
	int input3_w_stride = THCudaTensor_stride(state, input3,3);
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);

	
//	int flow1_grad_b_stride = THCudaTensor_stride(state,flow1_grad,0);
//	int flow1_grad_c_stride = THCudaTensor_stride(state,flow1_grad,1);
//	int flow1_grad_h_stride = THCudaTensor_stride(state,flow1_grad,2);
//	int flow1_grad_w_stride = THCudaTensor_stride(state,flow1_grad,3);
	
	int output_b_stride = THCudaTensor_stride(state,output,0);
	int output_c_stride = THCudaTensor_stride(state,output,1);
	int output_h_stride = THCudaTensor_stride(state,output,2);
	int output_w_stride = THCudaTensor_stride(state,output,3);	

	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
    if(input1_b_stride != THCudaTensor_stride(state, gradinput1,0)) return error;
	if(input2_b_stride != THCudaTensor_stride(state, gradinput2,0)) return error;
	if(input1_c_stride != THCudaTensor_stride(state, gradinput1,1)) return error;
	if(input2_c_stride != THCudaTensor_stride(state, gradinput2,1)) return error;
	if(input3_c_stride != THCudaTensor_stride(state, gradinput3,1)) return error;

    //printf("WeightLayer_gpu_backward GPU backward: input1 %d,%d,%d,%d\n",
    //    input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);
    //printf("WeightLayer_gpu_backward GPU backward: flow1_grad %d,%d,%d,%d\n",
	//		flow1_grad_b_stride,flow1_grad_c_stride,flow1_grad_h_stride,flow1_grad_w_stride);

	int	nElement = THCudaTensor_nElement(state, gradoutput);

	error  = WeightLayer_gpu_backward_kernel(
			THCState_getCurrentStream(state),
			nElement, //to let the nummous
			w,h,channel,batch,  

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
//			flow1_grad_b_stride,flow1_grad_c_stride,flow1_grad_h_stride,flow1_grad_w_stride,
			output_b_stride,  output_c_stride,   output_h_stride,   output_w_stride,

			THCudaTensor_data(state,input1),
			THCudaTensor_data(state,input2),
			THCudaTensor_data(state,input3),
//			THCudaTensor_data(state,flow1_grad),
			THCudaTensor_data(state,output),
			

			THCudaTensor_data(state,gradoutput),
			THCudaTensor_data(state,gradinput1),
			THCudaTensor_data(state,gradinput2),
			THCudaTensor_data(state,gradinput3),
//			THCudaTensor_data(state,gradflow1_grad),
			lambda_e,  lambda_v,   Nw
 
			);
	THCudaCheck(cudaGetLastError());

	return error;
}
		
int PixelValueLayer_gpu_forward(
		THCudaTensor * input1,  		THCudaTensor * input3, 	THCudaTensor * flow_weights, 		THCudaTensor * output,
		float sigma_d,    float tao_r ,  float Prowindow
		)
{
	int error = - 1;
	if(Prowindow != 2.0f) {printf("Prowindow must be odd number 2\n"); return error;}

	int channel = THCudaTensor_size(state, input1, 1);
	if(channel!=3) return error;
	int batch = THCudaTensor_size(state, input1,0);
	//if(THCudaTensor_size(state,input2, 0) != batch) return error;

	int h = THCudaTensor_size(state,input1,2);
	int w = THCudaTensor_size(state,input1,3);
	//if(THCudaTensor_size(state,input2,2) != h) return error;// to add some checkpoint
	//if(THCudaTensor_size(state,input2,3) != w) return error;

//    int filter_size2 = THCudaTensor_size(state, input3, 1);
  //  int filter_size = (int) sqrt((float) filter_size2);
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));
	if(THCudaTensor_size(state, input3, 1) != 2) {printf("the flow channel should be 2\n"); return error;}
	if(THCudaTensor_size(state, flow_weights,1) != 1){ printf("the flow_weights channel should be 1\n"); return error;}
	if(THCudaTensor_size(state, output,1) != 3) {printf("the image channel should be 3\n"); return error;}


	int input1_b_stride = THCudaTensor_stride(state, input1,0);
	int input1_c_stride = THCudaTensor_stride(state, input1,1);
	int input1_h_stride = THCudaTensor_stride(state, input1,2);
	int input1_w_stride = THCudaTensor_stride(state, input1,3);

	///int input2_b_stride = THCudaTensor_stride(state, input2,0);
	//int input2_c_stride = THCudaTensor_stride(state, input2,1);
	//int input2_h_stride = THCudaTensor_stride(state, input2,2);
	//int input2_w_stride = THCudaTensor_stride(state, input2,3);

    int input3_b_stride = THCudaTensor_stride(state, input3,0);
	int input3_c_stride = THCudaTensor_stride(state, input3,1);
	int input3_h_stride = THCudaTensor_stride(state, input3,2);
	int input3_w_stride = THCudaTensor_stride(state, input3,3);
	
	int flow_weights_b_stride = THCudaTensor_stride(state,flow_weights,0);
	int flow_weights_c_stride = THCudaTensor_stride(state,flow_weights,1);
	int flow_weights_h_stride = THCudaTensor_stride(state,flow_weights,2);
	int flow_weights_w_stride = THCudaTensor_stride(state,flow_weights,3);	
	
	int output_b_stride = THCudaTensor_stride(state,output,0);
	int output_c_stride = THCudaTensor_stride(state,output,1);
	int output_h_stride = THCudaTensor_stride(state,output,2);
	int output_w_stride = THCudaTensor_stride(state,output,3);	
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
    if(input1_w_stride !=1) return error;
	//if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
	if(flow_weights_w_stride !=1) return error;
	if(input1_b_stride != THCudaTensor_stride(state, output,0)) return error;
	if(input1_c_stride != THCudaTensor_stride(state, output,1)) return error;

	int	nElement = THCudaTensor_nElement(state, output);


	error = PixelValueLayer_gpu_forward_kernel(
			THCState_getCurrentStream(state),
			nElement,w,h,channel,batch,  

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			//input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			flow_weights_b_stride,flow_weights_c_stride,flow_weights_h_stride,flow_weights_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,

			THCudaTensor_data(state,input1),
			//THCudaTensor_data(state,input2),
			THCudaTensor_data(state,input3),
			THCudaTensor_data(state,flow_weights),
			THCudaTensor_data(state,output),
			
			sigma_d,      tao_r ,   Prowindow
			);
	THCudaCheck(cudaGetLastError());

	return error;		
}
int PixelValueLayer_gpu_backward(
		THCudaTensor * input1,  		THCudaTensor * input3, 	THCudaTensor * flow_weights, 		 
		THCudaTensor * gradoutput, 
		THCudaTensor * gradinput1,		THCudaTensor * gradinput3, THCudaTensor * gradflow_weights,
		float sigma_d,    float tao_r ,  float Prowindow
		)
{
	int error = - 1;
	if(Prowindow != 2.0f) {printf("Prowindow must be odd number 2\n"); return error;}

	int channel = THCudaTensor_size(state, input1, 1);
	if(channel!=3) return error;
	int batch = THCudaTensor_size(state, input1,0);
	//if(THCudaTensor_size(state,input2, 0) != batch) return error;
	//if(THCudaTensor_size(state, input2,1) != 2) return error;

	int h = THCudaTensor_size(state,input1,2);
	int w = THCudaTensor_size(state,input1,3);
	//if(THCudaTensor_size(state,input2,2) != h) return error;// to add some checkpoint
	//if(THCudaTensor_size(state,input2,3) != w) return error;


   // int filter_size2 = THCudaTensor_size(state, input3, 1);
    //int filter_size = (int) sqrt((float) filter_size2);
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));
	if(THCudaTensor_size(state, input3, 1) != 2) {printf("the flow channel should be 2\n"); return error;}
	if(THCudaTensor_size(state, flow_weights,1) != 1) {printf("the flow_weights channel should be 1\n"); return error;}
	//if(THCudaTensor_size(state, output,1) != 1){ printf("the flow weight channel should be 1\n"); return error;}

	int input1_b_stride = THCudaTensor_stride(state, input1,0);
	int input1_c_stride = THCudaTensor_stride(state, input1,1);
	int input1_h_stride = THCudaTensor_stride(state, input1,2);
	int input1_w_stride = THCudaTensor_stride(state, input1,3);

	//int input2_b_stride = THCudaTensor_stride(state, input2,0);
	//int input2_c_stride = THCudaTensor_stride(state, input2,1);
	//int input2_h_stride = THCudaTensor_stride(state, input2,2);
	//int input2_w_stride = THCudaTensor_stride(state, input2,3);
     
    int input3_b_stride = THCudaTensor_stride(state, input3,0);
	int input3_c_stride = THCudaTensor_stride(state, input3,1);
	int input3_h_stride = THCudaTensor_stride(state, input3,2);
	int input3_w_stride = THCudaTensor_stride(state, input3,3);
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	int flow_weights_b_stride = THCudaTensor_stride(state,flow_weights,0);
	int flow_weights_c_stride = THCudaTensor_stride(state,flow_weights,1);
	int flow_weights_h_stride = THCudaTensor_stride(state,flow_weights,2);
	int flow_weights_w_stride = THCudaTensor_stride(state,flow_weights,3);	
	
	int output_b_stride = THCudaTensor_stride(state,gradoutput,0);
	int output_c_stride = THCudaTensor_stride(state,gradoutput,1);
	int output_h_stride = THCudaTensor_stride(state,gradoutput,2);
	int output_w_stride = THCudaTensor_stride(state,gradoutput,3);		

	//TODO: do we need to assert the w_stride to be 1
	if(input1_w_stride !=1) return error;
	//if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
    if(input1_b_stride != THCudaTensor_stride(state, gradinput1,0)) return error;
	//if(input2_b_stride != THCudaTensor_stride(state, gradinput2,0)) return error;
	if(input1_c_stride != THCudaTensor_stride(state, gradinput1,1)) return error;
	//if(input2_c_stride != THCudaTensor_stride(state, gradinput2,1)) return error;
	if(input3_c_stride != THCudaTensor_stride(state, gradinput3,1)) return error;
	if(flow_weights_b_stride != THCudaTensor_stride(state, gradflow_weights,0)) return error;

//    printf("PixelValueLayer_gpu_backward GPU backward:  input1 %d,%d,%d,%d\n",
//        input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);
//    printf("PixelValueLayer_gpu_backward GPU backward: flow weights %d,%d,%d,%d\n",
//        THCudaTensor_stride(state, gradflow_weights,0),THCudaTensor_stride(state, gradflow_weights,1),
//        THCudaTensor_stride(state, gradflow_weights,2),THCudaTensor_stride(state, gradflow_weights,3));

	int	nElement = THCudaTensor_nElement(state, gradoutput);

	error  = PixelValueLayer_gpu_backward_kernel(
			THCState_getCurrentStream(state),
			nElement, //to let the nummous
			w,h,channel,batch,  

			input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			//input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			flow_weights_b_stride,flow_weights_c_stride,flow_weights_h_stride,flow_weights_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,


			THCudaTensor_data(state,input1),
			//THCudaTensor_data(state,input2),
			THCudaTensor_data(state,input3),
			THCudaTensor_data(state,flow_weights),
			//THCudaTensor_data(state,output),

			THCudaTensor_data(state,gradoutput),
			THCudaTensor_data(state,gradinput1),
			//THCudaTensor_data(state,gradinput2),
			THCudaTensor_data(state,gradinput3),
			THCudaTensor_data(state,gradflow_weights),
			sigma_d,      tao_r ,   Prowindow
 
			);
	THCudaCheck(cudaGetLastError());

	return error;		
}
int PixelWeightLayer_gpu_forward(
		THCudaTensor * input3, 	THCudaTensor * flow_weights, 		THCudaTensor * output,
		float sigma_d,    float tao_r ,  float Prowindow
		)
{
	int error = - 1;
	if(Prowindow != 2.0f) {printf("Prowindow must be odd number 2\n"); return error;}

	//int channel = THCudaTensor_size(state, input1, 1);
	//if(channel!=3) return error;
	int batch = THCudaTensor_size(state, input3,0);
	//if(THCudaTensor_size(state,input2, 0) != batch) return error;

	int h = THCudaTensor_size(state,input3,2);
	int w = THCudaTensor_size(state,input3,3);
	//if(THCudaTensor_size(state,input2,2) != h) return error;// to add some checkpoint
	//if(THCudaTensor_size(state,input2,3) != w) return error;

//    int filter_size2 = THCudaTensor_size(state, input3, 1);
  //  int filter_size = (int) sqrt((float) filter_size2);
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));
	if(THCudaTensor_size(state, input3, 1) != 2) {printf("the flow channel should be 2\n"); return error;}
	if(THCudaTensor_size(state, flow_weights,1) != 1){ printf("the flow_weights channel should be 1\n"); return error;}
	if(THCudaTensor_size(state, output,1) != 1) {printf("the flow weight channel should be 1\n"); return error;}


	//int input1_b_stride = THCudaTensor_stride(state, input1,0);
	//int input1_c_stride = THCudaTensor_stride(state, input1,1);
	//int input1_h_stride = THCudaTensor_stride(state, input1,2);
	//int input1_w_stride = THCudaTensor_stride(state, input1,3);

	///int input2_b_stride = THCudaTensor_stride(state, input2,0);
	//int input2_c_stride = THCudaTensor_stride(state, input2,1);
	//int input2_h_stride = THCudaTensor_stride(state, input2,2);
	//int input2_w_stride = THCudaTensor_stride(state, input2,3);

    int input3_b_stride = THCudaTensor_stride(state, input3,0);
	int input3_c_stride = THCudaTensor_stride(state, input3,1);
	int input3_h_stride = THCudaTensor_stride(state, input3,2);
	int input3_w_stride = THCudaTensor_stride(state, input3,3);
	
	int flow_weights_b_stride = THCudaTensor_stride(state,flow_weights,0);
	int flow_weights_c_stride = THCudaTensor_stride(state,flow_weights,1);
	int flow_weights_h_stride = THCudaTensor_stride(state,flow_weights,2);
	int flow_weights_w_stride = THCudaTensor_stride(state,flow_weights,3);	
	
	int output_b_stride = THCudaTensor_stride(state,output,0);
	int output_c_stride = THCudaTensor_stride(state,output,1);
	int output_h_stride = THCudaTensor_stride(state,output,2);
	int output_w_stride = THCudaTensor_stride(state,output,3);	
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
    //if(input1_w_stride !=1) return error;
	//if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
	if(flow_weights_w_stride !=1) return error;
	///if(input1_b_stride != THCudaTensor_stride(state, output,0)) return error;
	//if(input1_c_stride != THCudaTensor_stride(state, output,1)) return error;

	int	nElement = THCudaTensor_nElement(state, output);


	error = PixelWeightLayer_gpu_forward_kernel(
			THCState_getCurrentStream(state),
			nElement,w,h, batch,  

			//input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			//input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			flow_weights_b_stride,flow_weights_c_stride,flow_weights_h_stride,flow_weights_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,

			//THCudaTensor_data(state,input1),
			//THCudaTensor_data(state,input2),
			THCudaTensor_data(state,input3),
			THCudaTensor_data(state,flow_weights),
			THCudaTensor_data(state,output),
			
			sigma_d,      tao_r ,   Prowindow
			);
	THCudaCheck(cudaGetLastError());

	return error;				
}
int PixelWeightLayer_gpu_backward(
		THCudaTensor * input3, 	THCudaTensor * flow_weights, 	THCudaTensor * output,
		THCudaTensor * gradoutput, 
		THCudaTensor * gradinput3, THCudaTensor * gradflow_weights,
		float threshhold,
		float sigma_d,    float tao_r ,  float Prowindow
		)
{
	int error = - 1;
	if(Prowindow != 2.0f) {printf("Prowindow must be odd number 2\n"); return error;}

	//int channel = THCudaTensor_size(state, input1, 1);
	//if(channel!=3) return error;
	int batch = THCudaTensor_size(state, input3,0);
	//if(THCudaTensor_size(state,input2, 0) != batch) return error;
	//if(THCudaTensor_size(state, input2,1) != 2) return error;

	int h = THCudaTensor_size(state,input3,2);
	int w = THCudaTensor_size(state,input3,3);
	//if(THCudaTensor_size(state,input2,2) != h) return error;// to add some checkpoint
	//if(THCudaTensor_size(state,input2,3) != w) return error;


   // int filter_size2 = THCudaTensor_size(state, input3, 1);
    //int filter_size = (int) sqrt((float) filter_size2);
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));
	if(THCudaTensor_size(state, input3, 1) != 2) {printf("the flow channel should be 2\n"); return error;}
	if(THCudaTensor_size(state, flow_weights,1) != 1) {printf("the flow_weights channel should be 1\n"); return error;}
	//if(THCudaTensor_size(state, output,1) != 1){ printf("the flow weight channel should be 1\n"); return error;}

	//int input1_b_stride = THCudaTensor_stride(state, input1,0);
	//int input1_c_stride = THCudaTensor_stride(state, input1,1);
	//int input1_h_stride = THCudaTensor_stride(state, input1,2);
	//int input1_w_stride = THCudaTensor_stride(state, input1,3);

	//int input2_b_stride = THCudaTensor_stride(state, input2,0);
	//int input2_c_stride = THCudaTensor_stride(state, input2,1);
	//int input2_h_stride = THCudaTensor_stride(state, input2,2);
	//int input2_w_stride = THCudaTensor_stride(state, input2,3);
     
    int input3_b_stride = THCudaTensor_stride(state, input3,0);
	int input3_c_stride = THCudaTensor_stride(state, input3,1);
	int input3_h_stride = THCudaTensor_stride(state, input3,2);
	int input3_w_stride = THCudaTensor_stride(state, input3,3);
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	int flow_weights_b_stride = THCudaTensor_stride(state,flow_weights,0);
	int flow_weights_c_stride = THCudaTensor_stride(state,flow_weights,1);
	int flow_weights_h_stride = THCudaTensor_stride(state,flow_weights,2);
	int flow_weights_w_stride = THCudaTensor_stride(state,flow_weights,3);	
	
	int output_b_stride = THCudaTensor_stride(state,gradoutput,0);
	int output_c_stride = THCudaTensor_stride(state,gradoutput,1);
	int output_h_stride = THCudaTensor_stride(state,gradoutput,2);
	int output_w_stride = THCudaTensor_stride(state,gradoutput,3);		

	//TODO: do we need to assert the w_stride to be 1
	//if(input1_w_stride !=1) return error;
	//if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
    //if(input1_b_stride != THCudaTensor_stride(state, gradinput1,0)) return error;
	//if(input2_b_stride != THCudaTensor_stride(state, gradinput2,0)) return error;
	//if(input1_c_stride != THCudaTensor_stride(state, gradinput1,1)) return error;
	//if(input2_c_stride != THCudaTensor_stride(state, gradinput2,1)) return error;
	if(input3_c_stride != THCudaTensor_stride(state, gradinput3,1)) return error;

//    printf("GPU backward: %d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);

	int	nElement = THCudaTensor_nElement(state, gradoutput);

	error  = PixelWeightLayer_gpu_backward_kernel(
			THCState_getCurrentStream(state),
			nElement, //to let the nummous
			w,h, batch,  

			//input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			//input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			flow_weights_b_stride,flow_weights_c_stride,flow_weights_h_stride,flow_weights_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,


			//THCudaTensor_data(state,input1),
			//THCudaTensor_data(state,input2),
			THCudaTensor_data(state,input3),
			THCudaTensor_data(state,flow_weights),
			THCudaTensor_data(state,output),

			THCudaTensor_data(state,gradoutput),
			//THCudaTensor_data(state,gradinput1),
			//THCudaTensor_data(state,gradinput2),
			THCudaTensor_data(state,gradinput3),
			THCudaTensor_data(state,gradflow_weights),
			threshhold,
			sigma_d,      tao_r ,   Prowindow
 
			);
	THCudaCheck(cudaGetLastError());

	return error;			
}
		
//int ReliableValueLayer_gpu_forward(
//		THCudaTensor * input3, 	THCudaTensor * flow_weights, 		THCudaTensor * output,
//		float sigma_d,    float tao_r ,  float Prowindow
//		)
//{
//return 0;
//}
//int ReliableValueLayer_gpu_backward(
//		THCudaTensor * input3, 	THCudaTensor * flow_weights, 		THCudaTensor * output,
//		THCudaTensor * gradinput3, THCudaTensor * gradflow_weights,
//		float sigma_d,    float tao_r ,  float Prowindow
//	)
//{
//
//}
int ReliableWeightLayer_gpu_forward(
		THCudaTensor * input3,  THCudaTensor * output,
		float sigma_d,    float tao_r ,  float Prowindow
		)
{

	int error = - 1;
	if(Prowindow != 2.0f) {printf("Prowindow must be odd number 2\n"); return error;}

	//int channel = THCudaTensor_size(state, input1, 1);
	//if(channel!=3) return error;
	int batch = THCudaTensor_size(state, input3,0);
	//if(THCudaTensor_size(state,input2, 0) != batch) return error;

	int h = THCudaTensor_size(state,input3,2);
	int w = THCudaTensor_size(state,input3,3);
	//if(THCudaTensor_size(state,input2,2) != h) return error;// to add some checkpoint
	//if(THCudaTensor_size(state,input2,3) != w) return error;

//    int filter_size2 = THCudaTensor_size(state, input3, 1);
  //  int filter_size = (int) sqrt((float) filter_size2);
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));
	if(THCudaTensor_size(state, input3, 1) != 2) {printf("the flow channel should be 2\n"); return error;}
	//if(THCudaTensor_size(state, flow_weights,1) != 1){ printf("the flow_weights channel should be 1\n"); return error;}
	if(THCudaTensor_size(state, output,1) != 1) {printf("the flow weight channel should be 1\n"); return error;}


	//int input1_b_stride = THCudaTensor_stride(state, input1,0);
	//int input1_c_stride = THCudaTensor_stride(state, input1,1);
	//int input1_h_stride = THCudaTensor_stride(state, input1,2);
	//int input1_w_stride = THCudaTensor_stride(state, input1,3);

	///int input2_b_stride = THCudaTensor_stride(state, input2,0);
	//int input2_c_stride = THCudaTensor_stride(state, input2,1);
	//int input2_h_stride = THCudaTensor_stride(state, input2,2);
	//int input2_w_stride = THCudaTensor_stride(state, input2,3);

    int input3_b_stride = THCudaTensor_stride(state, input3,0);
	int input3_c_stride = THCudaTensor_stride(state, input3,1);
	int input3_h_stride = THCudaTensor_stride(state, input3,2);
	int input3_w_stride = THCudaTensor_stride(state, input3,3);
	
	//int flow_weights_b_stride = THCudaTensor_stride(state,flow_weights,0);
	//int flow_weights_c_stride = THCudaTensor_stride(state,flow_weights,1);
	//int flow_weights_h_stride = THCudaTensor_stride(state,flow_weights,2);
	//int flow_weights_w_stride = THCudaTensor_stride(state,flow_weights,3);	
	
	int output_b_stride = THCudaTensor_stride(state,output,0);
	int output_c_stride = THCudaTensor_stride(state,output,1);
	int output_h_stride = THCudaTensor_stride(state,output,2);
	int output_w_stride = THCudaTensor_stride(state,output,3);	
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//TODO: do we need to assert the w_stride to be 1
    //if(input1_w_stride !=1) return error;
	//if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
	//if(flow_weights_w_stride !=1) return error;
	///if(input1_b_stride != THCudaTensor_stride(state, output,0)) return error;
	//if(input1_c_stride != THCudaTensor_stride(state, output,1)) return error;

	int	nElement = THCudaTensor_nElement(state, output);


	error = ReliableWeightLayer_gpu_forward_kernel(
			THCState_getCurrentStream(state),
			nElement,w,h, batch,  

			//input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			//input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			//flow_weights_b_stride,flow_weights_c_stride,flow_weights_h_stride,flow_weights_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,

			//THCudaTensor_data(state,input1),
			//THCudaTensor_data(state,input2),
			THCudaTensor_data(state,input3),
			//THCudaTensor_data(state,flow_weights),
			THCudaTensor_data(state,output),
			
			sigma_d,      tao_r ,   Prowindow
			);
	THCudaCheck(cudaGetLastError());

	return error;		 
}
int ReliableWeightLayer_gpu_backward(
		THCudaTensor * input3, 	 THCudaTensor * output,
//		THCudaTensor * output,
		THCudaTensor * gradoutput, 
		THCudaTensor * gradinput3,
        float threshhold,
		float sigma_d,    float tao_r ,  float Prowindow
	)
{

	int error = - 1;
	if(Prowindow != 2.0f){ printf("Prowindow must be odd number 2\n"); return error;}

	//int channel = THCudaTensor_size(state, input1, 1);
	//if(channel!=3) return error;
	int batch = THCudaTensor_size(state, input3,0);
	//if(THCudaTensor_size(state,input2, 0) != batch) return error;
	//if(THCudaTensor_size(state, input2,1) != 2) return error;

	int h = THCudaTensor_size(state,input3,2);
	int w = THCudaTensor_size(state,input3,3);
	//if(THCudaTensor_size(state,input2,2) != h) return error;// to add some checkpoint
	//if(THCudaTensor_size(state,input2,3) != w) return error;


   // int filter_size2 = THCudaTensor_size(state, input3, 1);
    //int filter_size = (int) sqrt((float) filter_size2);
//    printf("filter size is: %d,or %f", filter_size, sqrt((float)filter_size2));
	if(THCudaTensor_size(state, input3, 1) != 2) {printf("the flow channel should be 2\n"); return error;}
//	if(THCudaTensor_size(state, flow_weights,1) != 1) printf("the flow_weights channel should be 1\n"); return error;
	//if(THCudaTensor_size(state, output,1) != 1) printf("the flow weight channel should be 1\n"); return error;

	//int input1_b_stride = THCudaTensor_stride(state, input1,0);
	//int input1_c_stride = THCudaTensor_stride(state, input1,1);
	//int input1_h_stride = THCudaTensor_stride(state, input1,2);
	//int input1_w_stride = THCudaTensor_stride(state, input1,3);

	//int input2_b_stride = THCudaTensor_stride(state, input2,0);
	//int input2_c_stride = THCudaTensor_stride(state, input2,1);
	//int input2_h_stride = THCudaTensor_stride(state, input2,2);
	//int input2_w_stride = THCudaTensor_stride(state, input2,3);
     
    int input3_b_stride = THCudaTensor_stride(state, input3,0);
	int input3_c_stride = THCudaTensor_stride(state, input3,1);
	int input3_h_stride = THCudaTensor_stride(state, input3,2);
	int input3_w_stride = THCudaTensor_stride(state, input3,3);
//    printf("filter tensor shape: %d,%d,%d,%d\n", input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride);


	//int flow_weights_b_stride = THCudaTensor_stride(state,flow_weights,0);
	//int flow_weights_c_stride = THCudaTensor_stride(state,flow_weights,1);
	//int flow_weights_h_stride = THCudaTensor_stride(state,flow_weights,2);
	//int flow_weights_w_stride = THCudaTensor_stride(state,flow_weights,3);	
	
	int output_b_stride = THCudaTensor_stride(state,gradoutput,0);
	int output_c_stride = THCudaTensor_stride(state,gradoutput,1);
	int output_h_stride = THCudaTensor_stride(state,gradoutput,2);
	int output_w_stride = THCudaTensor_stride(state,gradoutput,3);		

	//TODO: do we need to assert the w_stride to be 1
	//if(input1_w_stride !=1) return error;
	//if(input2_w_stride !=1) return error;
    if(input3_w_stride !=1) return error;
    //if(input1_b_stride != THCudaTensor_stride(state, gradinput1,0)) return error;
	//if(input2_b_stride != THCudaTensor_stride(state, gradinput2,0)) return error;
	//if(input1_c_stride != THCudaTensor_stride(state, gradinput1,1)) return error;
	//if(input2_c_stride != THCudaTensor_stride(state, gradinput2,1)) return error;
	if(input3_c_stride != THCudaTensor_stride(state, gradinput3,1)) return error;

//    printf("GPU backward: %d,%d,%d,%d\n", input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride);

	int	nElement = THCudaTensor_nElement(state, gradoutput);

	error  = ReliableWeightLayer_gpu_backward_kernel(
			THCState_getCurrentStream(state),
			nElement, //to let the nummous
			w,h, batch,  

			//input1_b_stride,input1_c_stride,input1_h_stride,input1_w_stride,
			//input2_b_stride,input2_c_stride,input2_h_stride,input2_w_stride,
			input3_b_stride,input3_c_stride,input3_h_stride,input3_w_stride,
			//flow_weights_b_stride,flow_weights_c_stride,flow_weights_h_stride,flow_weights_w_stride,
			output_b_stride,output_c_stride,output_h_stride,output_w_stride,


			//THCudaTensor_data(state,input1),
			//THCudaTensor_data(state,input2),
			THCudaTensor_data(state,input3),
			//THCudaTensor_data(state,flow_weights),
			THCudaTensor_data(state,output),

			THCudaTensor_data(state,gradoutput),
			//THCudaTensor_data(state,gradinput1),
			//THCudaTensor_data(state,gradinput2),
			THCudaTensor_data(state,gradinput3),
			//THCudaTensor_data(state,gradflow_weights),
			threshhold,
			sigma_d,      tao_r ,   Prowindow
			);
	THCudaCheck(cudaGetLastError());

	return error;			
}