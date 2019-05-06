
int SeparableConvFlowLayer_gpu_forward(
		THCudaTensor * input1,
		THCudaTensor * input2,
		THCudaTensor * input3,
		//THCudaTensor * output,
		THCudaTensor * flow_output
		);

int SeparableConvFlowLayer_gpu_backward(
		THCudaTensor * input1,
		THCudaTensor * input2,
		THCudaTensor * input3,
		THCudaTensor * gradflow_output,
		THCudaTensor * gradinput1,
		THCudaTensor * gradinput2,
		THCudaTensor * gradinput3
		); 
int SeparableConvLayer_gpu_forward(
		THCudaTensor * input1,
		THCudaTensor * input2,
		THCudaTensor * input3,
		THCudaTensor * output
		);

int SeparableConvLayer_gpu_backward(
		THCudaTensor * input1,
		THCudaTensor * input2,
		THCudaTensor * input3,
		THCudaTensor * gradoutput,
		THCudaTensor * gradinput1,
		THCudaTensor * gradinput2,
		THCudaTensor * gradinput3
		);


int InterpolationLayer_gpu_forward(
		THCudaTensor * input1,
		THCudaTensor * input2,
		THCudaTensor * output

		);

int InterpolationLayer_gpu_backward(
		THCudaTensor* input1,
		THCudaTensor* input2,

		THCudaTensor * gradoutput,

		THCudaTensor * gradinput1,
		THCudaTensor * gradinput2
		);
int InterpolationChLayer_gpu_forward(
		THCudaTensor * input1,
		THCudaTensor * input2,
		THCudaTensor * output

		);

int InterpolationChLayer_gpu_backward(
		THCudaTensor* input1,
		THCudaTensor* input2,

		THCudaTensor * gradoutput,

		THCudaTensor * gradinput1,
		THCudaTensor * gradinput2
		);
int FilterInterpolationLayer_gpu_forward(
		THCudaTensor * input1,
		THCudaTensor * input2,
		THCudaTensor * input3,
		THCudaTensor * output

		);

int FilterInterpolationLayer_gpu_backward(
		THCudaTensor * input1,
		THCudaTensor * input2,
		THCudaTensor * input3,
		THCudaTensor * gradoutput,
		THCudaTensor * gradinput1,
		THCudaTensor * gradinput2,
		THCudaTensor * gradinput3
		);

int FlowProjectionLayer_gpu_forward(
		THCudaTensor * input1,
		THCudaTensor * count,
		THCudaTensor * output,
		int fillhole
		);

int FlowProjectionLayer_gpu_backward(
		THCudaTensor * input1,
        THCudaTensor * count,
		THCudaTensor * gradoutput,
		THCudaTensor * gradinput1
		);

int DepthFlowProjectionLayer_gpu_forward(
		THCudaTensor * input1,
		THCudaTensor * input2,
		THCudaTensor * count,
		THCudaTensor * output,
		int fillhole
		);

int DepthFlowProjectionLayer_gpu_backward(
		THCudaTensor * input1,
		THCudaTensor * input2,
        THCudaTensor * count,
		THCudaTensor * output,
        THCudaTensor * gradoutput,
		THCudaTensor * gradinput1,
		THCudaTensor * gradinput2
		);

int WeightedFlowProjectionLayer_gpu_forward(
		THCudaTensor * input1,
		THCudaTensor * input2,
		THCudaTensor  * input3,
		THCudaTensor * count,
		THCudaTensor * weight,
		THCudaTensor * output,
		int fillhole,
		float threshhold
		);

int WeightedFlowProjectionLayer_gpu_backward(
		THCudaTensor * input1,
		THCudaTensor * input2,
		THCudaTensor * input3,
        THCudaTensor * count,
        THCudaTensor * weight,
		THCudaTensor * gradoutput,
		THCudaTensor * gradinput1,
		float threshhold
		);
//int FlowFillholelayer_gpu_forward(
//		THCudaTensor * input1,
//		THCudaTensor * output
//		);



int WeightLayer_gpu_forward(
		THCudaTensor * input1, 		THCudaTensor * input2, 		THCudaTensor * input3,
//		THCudaTensor * flow1_grad,
	    THCudaTensor * output,
		float  lambda_e, float	lambda_v, float Nw
		);
int WeightLayer_gpu_backward(
		THCudaTensor * input1, 		THCudaTensor * input2, 		THCudaTensor * input3,
//		 THCudaTensor * flow1_grad,
        THCudaTensor * output,
		THCudaTensor * gradoutput, 
		THCudaTensor * gradinput1,		THCudaTensor * gradinput2, 	THCudaTensor * gradinput3,
//		THCudaTensor *  gradflow1_grad,
		float  lambda_e, float	lambda_v, float Nw
		);
		
int PixelValueLayer_gpu_forward(
		THCudaTensor * input1,  THCudaTensor * input3, 	THCudaTensor * flow_weights, THCudaTensor * output,
		float sigma_d,    float tao_r ,  float Prowindow
		);
int PixelValueLayer_gpu_backward(
		THCudaTensor * input1,  		THCudaTensor * input3, 	THCudaTensor * flow_weights, 		 
		THCudaTensor * gradoutput, 
		THCudaTensor * gradinput1,		THCudaTensor * gradinput3, THCudaTensor * gradflow_weights,
		float sigma_d,    float tao_r ,  float Prowindow
		);
int PixelWeightLayer_gpu_forward(
		THCudaTensor * input3, 	THCudaTensor * flow_weights, 		THCudaTensor * output,
		float sigma_d,    float tao_r ,  float Prowindow
		);
int PixelWeightLayer_gpu_backward(
		THCudaTensor * input3, 	THCudaTensor * flow_weights,  THCudaTensor * output,
		THCudaTensor * gradoutput, 
		THCudaTensor * gradinput3, THCudaTensor * gradflow_weights,
        float threshhold,
		float sigma_d,    float tao_r ,  float Prowindow
		);
		
//int ReliableValueLayer_gpu_forward(
//		THCudaTensor * input3, 	THCudaTensor * flow_weights, 		THCudaTensor * output,
//		float sigma_d,    float tao_r ,  float Prowindow
//		); 
//int ReliableValueLayer_gpu_backward(
//		THCudaTensor * input3, 	THCudaTensor * flow_weights, 		THCudaTensor * output,
//		THCudaTensor * gradinput3, THCudaTensor * gradflow_weights,
//		float sigma_d,    float tao_r ,  float Prowindow
//	);
int ReliableWeightLayer_gpu_forward(
		THCudaTensor * input3,  THCudaTensor * output,
		float sigma_d,    float tao_r ,  float Prowindow
		); 
int ReliableWeightLayer_gpu_backward(
		THCudaTensor * input3, 	 	 THCudaTensor * output,
		THCudaTensor * gradoutput, 
		THCudaTensor * gradinput3,  
				float threshhold,
		float sigma_d,    float tao_r ,  float Prowindow
	); 