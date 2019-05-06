int SeparableConvFlowLayer_cpu_forward(
		THFloatTensor * input1,
		THFloatTensor * input2,
		THFloatTensor * input3,
		//THFloatTensor * output,
		THFloatTensor * flow_output
		);

int SeparableConvFlowLayer_cpu_backward(
		THFloatTensor * input1,
		THFloatTensor * input2,
		THFloatTensor * input3,
		THFloatTensor * gradflow_output,
		THFloatTensor * gradinput1,
		THFloatTensor * gradinput2,
		THFloatTensor * gradinput3
		);
int SeparableConvLayer_cpu_forward(
		THFloatTensor * input1,
		THFloatTensor * input2,
		THFloatTensor * input3,
		THFloatTensor * output
		);

int SeparableConvLayer_cpu_backward(
		THFloatTensor * input1,
		THFloatTensor * input2,
		THFloatTensor * input3,
		THFloatTensor * gradoutput,
		THFloatTensor * gradinput1,
		THFloatTensor * gradinput2,
		THFloatTensor * gradinput3
		);

int InterpolationLayer_cpu_forward(
		THFloatTensor * input1,
		THFloatTensor * input2,
		THFloatTensor * output
		);

int InterpolationLayer_cpu_backward(
		THFloatTensor * input1,
		THFloatTensor * input2,
		THFloatTensor * gradoutput,
		THFloatTensor * gradinput1,
		THFloatTensor * gradinput2
		);

int InterpolationChLayer_cpu_forward(
		THFloatTensor * input1,
		THFloatTensor * input2,
		THFloatTensor * output
		);

int InterpolationChLayer_cpu_backward(
		THFloatTensor * input1,
		THFloatTensor * input2,
		THFloatTensor * gradoutput,
		THFloatTensor * gradinput1,
		THFloatTensor * gradinput2
		);
int FilterInterpolationLayer_cpu_forward(
		THFloatTensor * input1,
		THFloatTensor * input2,
		THFloatTensor * input3,
		THFloatTensor * output
		);

int FilterInterpolationLayer_cpu_backward(
		THFloatTensor * input1,
		THFloatTensor * input2,
		THFloatTensor * input3,
		THFloatTensor * gradoutput,
		THFloatTensor * gradinput1,
		THFloatTensor * gradinput2,
		THFloatTensor * gradinput3
		);

int FlowProjectionLayer_cpu_forward(
		THFloatTensor * input1,
		THFloatTensor * count,
		THFloatTensor * output,
		int fillhole
		);

int FlowProjectionLayer_cpu_backward(
		THFloatTensor * input1,
        THFloatTensor * count,
        THFloatTensor * gradoutput,
		THFloatTensor * gradinput1
		);
int DepthFlowProjectionLayer_cpu_forward(
		THFloatTensor * input1,
		THFloatTensor * input2,
		THFloatTensor * count,
		THFloatTensor * output,
		int fillhole
		);

int DepthFlowProjectionLayer_cpu_backward(
		THFloatTensor * input1,
		THFloatTensor * input2,
        THFloatTensor * count,
        THFloatTensor * output,
        THFloatTensor * gradoutput,
		THFloatTensor * gradinput1,
		THFloatTensor * gradinput2
		);

int WeightedFlowProjectionLayer_cpu_forward(
		THFloatTensor * input1,
		THFloatTensor * input2,
		THFloatTensor * input3,
		THFloatTensor * count,
		THFloatTensor * weight,
		THFloatTensor * output,
		int fillhole,
		float threshold
		);

int WeightedFlowProjectionLayer_cpu_backward(
		THFloatTensor * input1,
        THFloatTensor * input2,
		THFloatTensor * input3,
        THFloatTensor * count,
        THFloatTensor * weight,
        THFloatTensor * gradoutput,
		THFloatTensor * gradinput1,
		float threshhold
		);

//int FlowFillholelayer_cpu_forward(
//		THFloatTensor * input1,
//		THFloatTensor * output
//		);
		
int WeightLayer_cpu_forward(
		THFloatTensor * input1, THFloatTensor * input2, THFloatTensor * input3,
//		THFloatTensor * flow1_grad,
		THFloatTensor * output,
		float  lambda_e, float	lambda_v, float Nw
		);
int WeightLayer_cpu_backward(
		THFloatTensor * input1, THFloatTensor * input2, THFloatTensor * input3,
//		THFloatTensor * flow1_grad,
        THFloatTensor * output,
		THFloatTensor * gradoutput, 
		THFloatTensor * gradinput1,		THFloatTensor * gradinput2,
		THFloatTensor * gradinput3,
//		THFloatTensor *  gradflow1_grad,
		float  lambda_e, float	lambda_v, float Nw
		);
		
int PixelValueLayer_cpu_forward(
		THFloatTensor * input1,  THFloatTensor * input3, THFloatTensor * flow_weights,THFloatTensor * output,
		float sigma_d,    float tao_r ,  float Prowindow
		);
int PixelValueLayer_cpu_backward(
		THFloatTensor * input1,  		THFloatTensor * input3, 	THFloatTensor * flow_weights, 		 
		THFloatTensor * gradoutput, 
		THFloatTensor * gradinput1,		THFloatTensor * gradinput3, THFloatTensor * gradflow_weights,
		float sigma_d,    float tao_r ,  float Prowindow
		);
int PixelWeightLayer_cpu_forward(
		THFloatTensor * input3, 	THFloatTensor * flow_weights, 		THFloatTensor * output,
		float sigma_d,    float tao_r ,  float Prowindow
		);
int PixelWeightLayer_cpu_backward(
		THFloatTensor * input3, 	THFloatTensor * flow_weights, 	THFloatTensor * output,
		THFloatTensor * gradoutput, 
		THFloatTensor * gradinput3, THFloatTensor * gradflow_weights,
        float threshhold,
		float sigma_d,    float tao_r ,  float Prowindow
		);
		
//int ReliableValueLayer_cpu_forward(
//		THFloatTensor * input3, 	THFloatTensor * flow_weights, 		THFloatTensor * output,
//		float sigma_d,    float tao_r ,  float Prowindow
//		); 
//int ReliableValueLayer_cpu_backward(
//		THFloatTensor * input3, 	THFloatTensor * flow_weights, 		THFloatTensor * output,
//		THFloatTensor * gradinput3, THFloatTensor * gradflow_weights,
//		float sigma_d,    float tao_r ,  float Prowindow
//	);
int ReliableWeightLayer_cpu_forward(
		THFloatTensor * input3,  THFloatTensor * output,
		float sigma_d,    float tao_r ,  float Prowindow
		); 
int ReliableWeightLayer_cpu_backward(
		THFloatTensor * input3, 	  THFloatTensor * output,
		THFloatTensor * gradoutput, 
		THFloatTensor * gradinput3,
		float threshhold,
		float sigma_d,    float tao_r ,  float Prowindow
	);
		