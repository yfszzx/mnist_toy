 

__global__ void g_L1_weight_deriv(float *weight,float *deriv,float param,int len){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if(idx<len){
		short y;
		float x;
		x=weight[idx];
		if(x!=0)y=(x>0)?-1:1;
		else y=0;
		deriv[idx]+=param*y;
	}
}
__global__ void g_add_threshold(float *w,float *o,int nodes_num,int data_num){
	int node_idx=blockIdx.x;
	int data_idx=blockIdx.y*blockDim.x+threadIdx.x;
	__shared__ float t;
	if(threadIdx.x==0)t=w[node_idx];
	__syncthreads();
	if(data_idx<data_num){
		o[data_idx*nodes_num+node_idx]+=t;
	}
	
}
struct gpu_top_sigmoid_deriv{
__device__ float operator ()( float & top,float & out) const {  
		float v=out;
		return top*v*(1.0f-v);
	}
}  ;
void g_top_sigmoid_deriv(float *top,float *out,int len){
	thrust :: device_ptr <float> t(top); 
	thrust :: device_ptr <float> o(out); 
	thrust :: transform (t, t+len ,o,o ,  gpu_top_sigmoid_deriv());
	//CUDA_CHECK;
}
struct gpu_top_tanh_deriv{
__device__ float operator ()(float & top,float & out) const {  
		float v=out;
		return top*(1.0f-v*v);
	}
}  ;
void g_top_tanh_deriv(float *top,float *out,int len){
	thrust :: device_ptr <float> t(top); 
	thrust :: device_ptr <float> o(out); 
	thrust :: transform (t, t+len ,o,o ,  gpu_top_tanh_deriv());
	//CUDA_CHECK;
}
struct gpu_sigmoid {  
__device__ float operator ()( float & x) const {  
return 1.0f/(1.0f+__expf(-x));
}  
};
struct gpu_tanh {  
__device__ float operator ()( float & x) const {  
return 1.0f-2.0f/(1.0f+__expf(2*x));; 
}  
}; 
void g_sigmoid(float *arr,int dimen){
	thrust :: device_ptr <float> f(arr); 
	thrust :: transform (f, f+dimen ,f , gpu_sigmoid());
	
}
void g_tanh(float *arr,int dimen){
	thrust :: device_ptr <float> f(arr); 
	thrust :: transform (f, f+dimen ,f , gpu_tanh());
}
class perceptrons{
private:
	array_group_sum *a_sum;
	float *tmp_array;	
	int len;
	void free_mem(){
		if(output!=NULL){
			cudaFree(output);
			cudaFree(tmp_array);			
			CUDA_CHECK;	
			delete a_sum;
		}
		output=NULL;
	}
	int block_y;
	float alpha;
	float beta;
	float a_alpha;
public:
	int input_dimen;
	int nodes_num;
	char type;//l:line,t:tanh,s:sigmoid
	int data_num;
	int weight_len; 
	float *weight;
	float *output;	
	float *deriv;
	float decay;
	char decay_mod;
	perceptrons(){
		nodes_num=0;
		data_num=0;
		decay=0;
	    decay_mod='0';
		alpha=1.0f;
		beta=0;
	}
	~perceptrons(){
		free_mem();
	}
	void read_struct(ifstream *fin){
		fin->read((char *)&input_dimen,sizeof(int));
		fin->read((char *)&nodes_num,sizeof(int));
		fin->read((char *)&decay_mod,sizeof(char));
		fin->read((char *)&decay,sizeof(float));
		fin->read((char *)&type,sizeof(char));
	}
	void write_struct(ofstream *fin){
		fin->write((char *)&input_dimen,sizeof(int));
		fin->write((char *)&nodes_num,sizeof(int));
		fin->write((char *)&decay_mod,sizeof(char));
		fin->write((char *)&decay,sizeof(float));
		fin->write((char *)&type,sizeof(char));
	}		
	void init_struct(){
		weight_len=(input_dimen+1)*nodes_num;	
		output=NULL;
	}
	void init_struct(int dimen,int nodes,char tp){
		input_dimen=dimen;
		nodes_num=nodes;
		type=tp;
		init_struct();
	}

	void show_struct(int idx){
		coutd<<"<layer "<<idx<<">";
		if(nodes_num>0){
			if(cout_show)cout<<"  input:"<<input_dimen<<"  nodes:"<<nodes_num<<"  type:"<<type<<"  param_num:"<<weight_len;
			coutd;
			memery();
			show_memery_size(memery(),'g',"ռ��");
		}else cout<<"  ��δ��ʼ��";
	}
	int memery(){
		int ret=sizeof(float)*input_dimen*data_num+sizeof(float)*nodes_num*data_num+sizeof(float)*nodes_num*data_num;
		return ret;
	}	
	void set_data_num(int num){
		if(num==data_num)return;
		free_mem();
		data_num=num;
		len=input_dimen*data_num;
		cudaMalloc((void**)&output,sizeof(float)*nodes_num*data_num);
		cudaMalloc((void**)&tmp_array,sizeof(float)*len);
		CUDA_CHECK;
		block_y=(data_num+g_threads-1)/g_threads;
		a_sum=new array_group_sum(nodes_num,data_num);		
		a_alpha=1.0f/data_num;
	}
	void run(float *input){
		CUBT(cublasSgemm(cublasHandle,CUBLAS_OP_N,CUBLAS_OP_N,
			nodes_num,data_num,input_dimen, &alpha,weight,nodes_num,input,input_dimen,&beta,output,nodes_num));
		dim3 blk(nodes_num,block_y);
		g_add_threshold<<<blk,g_threads>>>(weight+input_dimen*nodes_num,output,nodes_num,data_num);		
		CUDA_CHECK;
		if(type=='t')g_tanh(output,nodes_num*data_num);
		if(type=='s')g_sigmoid(output,nodes_num*data_num);	
		//coutd<<"oo"<<array_length(output,nodes_num*data_num);
	}	
	void weight_rand(){
			float *tmp=new float[weight_len];
			for(int i=0;i<weight_len;i++){ //��ʼ��Χ�ڣ�-0.01,0.01��֮��
				tmp[i]=float(rand())/RAND_MAX/100*(rand()%2*2-1);					
			}
			cudaMemcpy(weight,tmp,sizeof(float)*weight_len,cudaMemcpyHostToDevice);			
			CUDA_CHECK;
			delete [] tmp;
	}
	void get_deriv(float *input){
		//coutd<<"d0"<<array_length(deriv,weight_len);
		//coutd<<"o"<<array_length(output,nodes_num*data_num);
		//coutd<<"i"<<array_length(input,input_dimen*data_num);
		CUBT(cublasSgemm(cublasHandle, CUBLAS_OP_N,	CUBLAS_OP_T,
			nodes_num,input_dimen,data_num, &alpha,output,nodes_num,input,input_dimen,&beta,deriv,nodes_num));
		//coutd<<"o"<<array_length(output,nodes_num*data_num);
		//��ֵ��ƫ��
		//coutd<<"d1"<<array_length(deriv,weight_len);
		//coutd<<"o"<<array_length(output,nodes_num*data_num);
		a_sum->sum(deriv+input_dimen*nodes_num,output);
		
		//coutd<<"o"<<array_length(output,nodes_num*data_num);
		//ȡƽ��
		CUBT(cublasSscal(cublasHandle,weight_len, &a_alpha,deriv, 1));
		//coutd<<"d2"<<array_length(deriv,weight_len);getchar();

	}
	void get_sub_deriv(float *sub_deriv,char sub_type){
		if(sub_type=='t'||sub_type=='s'){
			//coutd<<"w"<<array_length(weight,weight_len);
			//coutd<<"os "<<array_length(sub_deriv,input_dimen*data_num);
			//coutd<<"oo"<<array_length(output,nodes_num*data_num);
			cudaMemset(tmp_array,0,sizeof(float)*len);
			//coutd<<"t"<<array_length(tmp_array,len);
			CUDA_CHECK;
			CUBT(cublasSgemm(cublasHandle, CUBLAS_OP_N,CUBLAS_OP_N,
				input_dimen,data_num,nodes_num, &alpha,weight,input_dimen,output,nodes_num,&beta,tmp_array,input_dimen
			));
	
			//coutd<<"t"<<array_length(tmp_array,len);
			if(sub_type=='t')g_top_tanh_deriv(tmp_array,sub_deriv,len);
			if(sub_type=='s')g_top_sigmoid_deriv(tmp_array,sub_deriv,len);
			//coutd<<"s"<<array_length(tmp_array,len);
			//coutd<<"oss "<<array_length(sub_deriv,input_dimen*data_num);
		}
		if(sub_type=='l'){
			CUBT(cublasSgemm(cublasHandle, CUBLAS_OP_N,	CUBLAS_OP_N,
				input_dimen,data_num,nodes_num, &alpha,weight,input_dimen,output,nodes_num,&beta,sub_deriv,input_dimen
			));
		}
	}

	float weight_decay_sum(){
		if(decay==0)return 0;
		float ret=0;
		if(decay_mod=='2'){
			CUBT(cublasSnrm2(cublasHandle,weight_len-nodes_num,weight,1,&ret));
			ret*=ret;
		}
		if(decay_mod=='1')
			CUBT(cublasSasum(cublasHandle,weight_len-nodes_num,weight,1,&ret));		
		return ret*decay;
	}
	void get_decay_deriv(){
		/*�����㷧ֵ��˥��*/
		if(decay==0)return;
		if(decay_mod=='2'){
			float t=2*decay;
			CUBT(cublasSaxpy(cublasHandle,weight_len-nodes_num, &t,weight, 1,deriv, 1));
		}
		if(decay_mod=='1'){
			int blocks=(weight_len-nodes_num+g_threads-1)/g_threads;
			g_L1_weight_deriv<<<blocks,g_threads>>>(weight,deriv,decay,len);
		}
	}
};

