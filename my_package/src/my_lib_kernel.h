
#ifdef __cplusplus
extern "C" {
#endif

int SeparableConvFlowLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch, const int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
	//	const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,
		const int flow_output_b_stride, const int flow_output_c_stride, const int flow_output_h_stride, const int flow_output_w_stride,

		const float* input1,    		const float* input2,    	const float* input3, 	 float* flow_output

		);

int SeparableConvFlowLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    		const int channel,  		const int batch, const int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
	//	const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,
		const int flow_output_b_stride, const int flow_output_c_stride, const int flow_output_h_stride, const int flow_output_w_stride,

		const float* input1,        		const float* input2,		const float* input3,

		const float* gradflow_output,    		float* gradinput1,  		float* gradinput2,  		float* gradinput3
		);
		
int SeparableConvLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch, const int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		const float* input1,    		const float* input2,    	const float* input3, 	float* output

		);

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
		);


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

		);

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
		);


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

		);

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
		);
int FilterInterpolationLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch, const  int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,

		const float* input1,    		const float* input2,    	const float* input3, 	float* output

		);

int FilterInterpolationLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    		const int channel,  		const int batch,    		const int filter_size,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,

		const float* input1,        		const float* input2,		const float* input3,

		const float* gradoutput,    		float* gradinput1,  		float* gradinput2,  		float* gradinput3
		);


int FlowProjection_gpu_forward_kernel(
		cudaStream_t stream, 		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch, const int fillhole,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,

		const float* input1,
		float* count,
		float* output

		);

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
		float* gradinput1
		);

int DepthFlowProjection_gpu_forward_kernel(
		cudaStream_t stream, 		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch, const int fillhole,
		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
        const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
        const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,

		const float* input1,		const float* input2,
		float* count,
		float* output

		);

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

		const float* input1,
		const float* input2,
        const float* count,
        const float* output,
		const float* gradoutput,
		float* gradinput1,
		float* gradinput2
		);

int WeightedFlowProjection_gpu_forward_kernel(
		cudaStream_t stream, 		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch, const int fillhole, const float threshhold,

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
        const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
        const int count_b_stride, const int count_c_stride, const int count_h_stride, const int count_w_stride,
		const int weight_b_stride,const int weight_c_stride, const int weight_h_stride, const int weight_w_stride,

		const float* input1,const float* input2, const float* input3,
		float* count, float *weight,
		float* output

		);

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
		const float* count,const float * weight,
		const float* gradoutput,
		float* gradinput1
		);

int WeightLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 		const int channel, 		const int batch,  

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
//		const int flow1_grad_b_stride,const int flow1_grad_c_stride,const int flow1_grad_h_stride,const int flow1_grad_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		const float* input1,    		const float* input2,    	const float* input3,
//		const float * flow1_grad,
        float* output,
		float  lambda_e, float	lambda_v, float Nw
		);

int WeightLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    		const int channel,  		const int batch,    		 

		const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
//		const int flow1_grad_b_stride,const int flow1_grad_c_stride,const int flow1_grad_h_stride,const int flow1_grad_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		const float* input1,     	const float* input2, const float* input3,
//		const float * flow1_grad,
        const float* output,

		const float* gradoutput, float* gradinput1, float* gradinput2, float* gradinput3,
//		 float* gradflow1_grad,

		float  lambda_e, float	lambda_v, float Nw

		);


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
		);

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

		);		


int PixelWeightLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 		 	const int batch,  

		//const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		const float* input3, const float * flow_weights,	float* output,
		float	sigma_d,     float tao_r , float  Prowindow
		);

int PixelWeightLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    	 		const int batch,    		 

		//const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,
		
		const float* input3, const float * flow_weights,const	 float* output,
		const float* gradoutput,  float* gradinput3, float* gradflow_weights,
        float threshhold,
		float	sigma_d,     float tao_r , float  Prowindow

		);

int ReliableWeightLayer_gpu_forward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w, 		const int h, 	 		const int batch,  

		//const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		//const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,

		const float* input3,  	float* output,
		float	sigma_d,     float tao_r , float  Prowindow
		);

int ReliableWeightLayer_gpu_backward_kernel(
		cudaStream_t stream,
		const int nElement,
		const int w,    		const int h,    	 		const int batch,    		 

		//const int input1_b_stride, const int input1_c_stride, const int input1_h_stride, const int input1_w_stride,
		//const int input2_b_stride, const int input2_c_stride, const int input2_h_stride, const int input2_w_stride,
		const int input3_b_stride, const int input3_c_stride, const int input3_h_stride, const int input3_w_stride,
		//const int flow_weights_b_stride,const int flow_weights_c_stride,const int flow_weights_h_stride,const int flow_weights_w_stride,
		const int output_b_stride, const int output_c_stride, const int output_h_stride, const int output_w_stride,
		
		const float* input3,   
        const float* output,
		const float* gradoutput,  float* gradinput3,
        float threshhold,
		float	sigma_d,     float tao_r , float  Prowindow

		);
		
#ifdef __cplusplus
}
#endif

