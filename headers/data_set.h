template <typename DATA_CLASS>
class load_train_data_class{
public:
	struct data{
		float **input;
		float **target;
		float *input_data;
		float *target_data;
		float compare_value;
		int data_num;
		int memsize;
		int input_dimen;
		int output_dimen;
	};
	int input_dimen;
	int output_dimen;
	float distortion_scale;
	float noise_scale;
	float * input,target;
	DATA_CLASS *data_set;
	data trn,cmp;
	load_train_data_class(DATA_CLASS *p){
		input_dimen=p->input_dimen;
		output_dimen=p->output_dimen;
		input=NULL;
		target=NULL;
		data_set=p;
		distortion_scale=0;
		noise_scale=0;
	 }
	~load_train_data_class(){
		if(input!=NULL)delete [] input;
		if(target!=NULL)delete [] target;
	}
	void pre_load(char mod){
		 data_num=(mod=='c')?cmp.data_num:trn.data_num;
		 load_mod=mod;
		 if(input!=NULL)delete [] input;
		 if(target!=NULL)delete [] target;
		 input=new float[input_dimen*data_num];
		 target=new float[output_dimen*data_num];
		 data_set->get_data(load_mod,input,target,data_num,distortion_scale);		
	 }
	void load_data(){
		dt.malloc(data_num,load_mod);
		if(load_mod=='t'){
			trn_noise.malloc(dt_num);
		}
		cudaMemcpy(dt.input,input,sizeof(float)*input_dimen*dt.data_num,cudaMemcpyHostToDevice);
		CUDA_CHECK;
		cudaMemcpy(dt.target,target,sizeof(float)*output_dimen*dt.data_num,cudaMemcpyHostToDevice);
		CUDA_CHECK;		
		if(load_mod=='t')trn_noise.add_noise(noise_scale);
	}
	float distortion_scale;
	void * data_set;
	void * deep_train_class;
	 void get_data();
	 void get_sub_data();
	 void create_bagging(int);
	 void pre_load(char mod,int num){
		 load_mod=mod;
		 data_num=num;
		 if(input!=NULL)delete [] input;
		 if(target!=NULL)delete [] target;
		 input=new float[input_dimen*data_num];
		 target=new float[output_dimen*data_num];
		 get_data();		
	 }
	 void pre_load(char mod,char *path_i,char *path_l){
		ifstream f(path_i,ios::binary);
		if(!f){
			cout<<endl<<"没有找到对照数据集";
			return ;
		}
		cout<<endl<<"载入对照输入数据集"<<path_i<<endl;
		load_mod=mod;
		if(input!=NULL)delete [] input;
		if(target!=NULL)delete [] target;
		int t;
		f.read((char *)&t,sizeof(int));
		f.read((char *)&data_num,sizeof(int));
		input=new float[input_dimen*data_num];
		target=new float[output_dimen*data_num];
		f.read((char *)input,sizeof(float)*input_dimen*data_num);
		f.close();
		f.open(path_l,ios::binary);
		f.read((char *)&t,sizeof(int));
		f.read((char *)&t,sizeof(int));			
		f.read((char *)target,sizeof(float)*output_dimen*data_num);
		f.close();		
	 }
	  void sub_pre_load(char mod,int num){
		 load_mod=mod;
		 data_num=num;
		 if(input!=NULL)delete [] input;
		 if(target!=NULL)delete [] target;
		 input=new float[input_dimen*data_num];
		 target=new float[output_dimen*data_num];
		 get_sub_data();		
	 }
	 float *input;
	 float *target;
	 int input_dimen;
	 int output_dimen;
	 int data_num;
	 char load_mod;
	 load_train_data_class(int i_dimen,int o_dimen){
		input_dimen=i_dimen;
		output_dimen=o_dimen;
		input=NULL;
		target=NULL;
	 }
	~load_train_data_class(){
		if(input!=NULL)delete [] input;
		if(target!=NULL)delete [] target;
	}
};

class data_set{
	train_set_struct train_set;
	struct data{
		float **input;
		float **target;
		float *input_data;
		float *target_data;
		float compare_value;
		int data_num;
		int memsize;
		int input_dimen;
		int output_dimen;
		data(){
			input_data=NULL;
			target_data=NULL;
			input=NULL;
			target=NULL;
			data_num=0;
			memsize=0;
			compare_value;
		}
		void malloc(int d_num,int i_dimen,int o_dimen,char train_mod){
			input_dimen=i_dimen;
			output_dimen=o_dimen;
			if(input_data!=NULL){
				memsize-=data_num*input_dimen*sizeof(float);
				cudaFree(input_data);
				CUDA_CHECK;
				if(train_mod!='l'){
					memsize-=data_num*output_dimen*sizeof(float);
					cudaFree(target_data);
					CUDA_CHECK;

				}

				delete [] input;
				delete [] target;
			}
			data_num=d_num;
			cudaMalloc((void**)&input_data,sizeof(float)*d_num*input_dimen);
			memsize+=d_num*input_dimen*sizeof(float);
			CUDA_CHECK;
			if(train_mod!='l'){
				cudaMalloc((void**)&target_data,sizeof(float)*d_num*output_dimen);
				memsize+=d_num*output_dimen*sizeof(float);	
				CUDA_CHECK;
			}
			else target_data=input_data;
			input=new float *[d_num];
			target=new float *[d_num];
			for(int i=0;i<d_num;i++){
				input[i]=input_data+i*input_dimen;
				target[i]=target_data+i*output_dimen;
			}
		}

		void pre(char top_type){
			/*if(top_type=='t'||top_type=='s'||top_type=='r'){
				int len=output_dimen*data_num;
				int blocks=(len+g_threads-1)/g_threads;
				if(top_type=='s')gpu_sigmoid<<<blocks,g_threads>>>(target[0],len);
				else gpu_tanh<<<blocks,g_threads>>>(target[0],len);
				CUDA_CHECK;
			}*/
			compare_value=0;
			float *one;
			cudaMalloc((void**)&one,sizeof(float)*data_num);		
			CUDA_CHECK;
			for(int j=0;j<output_dimen;j++){
				float target_avg=g_sum_value(target[0]+j,data_num,output_dimen);
				target_avg/=-data_num;
				g_set_one(one,data_num);
				CUBT(cublasSscal(cublasHandle,data_num, &target_avg,one, 1));
				float x=1;
				CUBT(cublasSaxpy (cublasHandle,data_num,&x,target[0]+j,output_dimen,one,1));
				CUBT(cublasSnrm2 (cublasHandle,data_num,one, 1,&x));
				x*=x;
				compare_value+=x/data_num;
			}
			cudaFree(one);
			CUDA_CHECK;
		}		
	};	
	struct noise_struct{
		float *scale;
		int dimen;
		void init(int input){
			dimen=input;
			scale=new float[dimen];
		}
	void get_scale(float **data,int data_num){
			float *one;
			float *tmp;
			cudaMalloc((void**)&one,sizeof(float)*data_num);
			CUDA_CHECK;
			cudaMalloc((void**)&tmp,sizeof(float)*data_num);
			CUDA_CHECK;
			g_set_one(one,data_num);
			float avg;
			for(int i=0;i<dimen;i++){
				CUBT(cublasScopy(cublasHandle,data_num,data[0]+i,dimen,tmp,1));
				CUBT(cublasSdot (cublasHandle,data_num, one, 1, tmp,1, &avg));
				avg/=-data_num;			
				CUBT(cublasSaxpy(cublasHandle,data_num,&avg, one, 1,tmp, 1));
				CUBT(cublasSnrm2(cublasHandle,data_num, tmp, 1, scale+i));
				scale[i]/=sqrt((float)data_num);
			}
			cudaFree(one);
			cudaFree(tmp);
		};
		void add(float **data,float **noise_data,int data_num,float noise_scale){
			if(noise_scale==0)return;
			float *rnd;

			CUDA_CHECK;
			curandGenerator_t gen;
			cudaMalloc( (void **) &rnd, (data_num+1)* sizeof(float) ) ;
			CHECK_CURAND( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
			CHECK_CURAND( curandSetPseudoRandomGeneratorSeed(gen,time(NULL)+rand()));
			float tmp=1.0f;
			CUBT(cublasScopy(cublasHandle,data_num*dimen,data[0],1,noise_data[0],1));
			for(int i=0;i<dimen;i++){
				if(scale[i]>0){
					CHECK_CURAND( curandGenerateNormal(gen, rnd,(data_num%2==1)?(data_num+1):data_num,0,scale[i]*noise_scale) );
					CUBT(cublasSaxpy(cublasHandle,data_num,&tmp, rnd, 1,noise_data[0]+i, dimen));
				}
			}
			cudaFree(rnd);
			CUDA_CHECK;
			CHECK_CURAND( curandDestroyGenerator(gen) );


		}

	};
	noise_struct noise;
	void init_struct(){	
		if(weight!=NULL){
			cudaFree(weight);
			memsize.strct-=sizeof(float)*weight_len;	
		}
		weight_len=0;
		for(int i=0;i<layers_num;i++){
			nervs[i]->init_struct();
			weight_len+=nervs[i]->weight_len;
		}
		cudaMalloc((void**)&weight,sizeof(float)*weight_len);		
		CUDA_CHECK;		
		memsize.strct+=sizeof(float)*weight_len;	
		if(train_mod!='r'){
			if(deriv!=NULL){
				cudaFree(deriv);
				memsize.strct-=sizeof(float)*weight_len;	
			}
			cudaMalloc((void**)&deriv,sizeof(float)*weight_len);
			CUDA_CHECK;
			memsize.strct+=sizeof(float)*weight_len;
			search_tool::pos=weight;
			search_tool::deriv=deriv;	
		}
		int p=0;
		for(int i=0;i<layers_num;i++){
			if(train_mod=='r')nervs[i]->init_weight(weight+p);
			else nervs[i]->init_weight(weight+p,deriv+p);
			p+=nervs[i]->weight_len;
		}		
	}
	void weight_rand(){
		for(int i=0;i<layers_num;i++)
			nervs[i]->weight_rand();
	}
	void set_data_num(int dt_num){
		if(dt_num==data_num)return;
		if(output_tmp!=NULL)cudaFree(output_tmp);
		data_num=dt_num;
		memsize.strct=0;
		for(int i=0;i<layers_num;i++){
			nervs[i]->set_data_num(data_num);
			memsize.strct+=nervs[i]->memsize;
		}
		cudaMalloc((void**)&output_tmp,sizeof(float)*output_dimen*data_num);
		memsize.strct+=sizeof(float)*output_dimen*data_num;
	}
	float run(float **input,float **target,int num=-1){			
		if(num==-1)num=data_num;  
		nerv_bottom->run(input,num);
		for(int i=1;i<layers_num;i++){
			nervs[i]->run(nervs[i-1]->output,num);
		}
		float *o;
		int dimen=output_dimen*num;
		cudaMalloc((void**)&o,sizeof(float)*dimen);
		CUDA_CHECK;
		int blocks=(dimen+g_threads-1)/g_threads;
		//float dev=search_tool::dev(nerv_top->output[0],output_dimen*num);
		gpu_loss_function<<<blocks,g_threads>>>(nerv_top->output[0],target[0],o,output_tmp,dimen,0,train_set.loss_mod,output_dimen);
		
		float ret=g_sum_value(o,dimen)/num;
		cudaFree(o);
		CUDA_CHECK;
		return ret;	
	}
	void run(float *s_input,float *out,int out_layer=-1,int num=-1,bool in_cuda_pos=false,bool out_cuda_pos=false){
		if(num==-1)num=data_num;
		if(out_layer==-1)out_layer=layers_num;
		float *input;
		if(in_cuda_pos){
			input=s_input;
		}else{
			cudaMalloc((void**)&input,sizeof(float)*input_dimen*num);
			CUDA_CHECK;
			cudaMemcpy(input,s_input,sizeof(float)*input_dimen*num,cudaMemcpyHostToDevice);
			CUDA_CHECK;

		}
		nerv_bottom->run(&input,num);
		for(int i=1;i<out_layer;i++){
			nervs[i]->run(nervs[i-1]->output,num);
		}	
		cudaMemcpy(out,nervs[out_layer-1]->output_data,sizeof(float)*nervs[out_layer-1]->nodes_num*num,((out_cuda_pos)?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));		
		CUDA_CHECK;
		if(!in_cuda_pos){
			cudaFree(input);
			CUDA_CHECK;
		}
	}
	void top_pre_deriv(double param,float **target){
		float t=2.0f;
		int len=output_dimen*data_num;
		//if(train_set.loss_mod=='2')CUBT(cublasSscal(cublasHandle,len, &t,output_tmp, 1));
		if(nerv_top->type=='t'){
			int blocks=(len+g_threads-1)/g_threads;
			g_top_tanh_deriv<<<blocks,g_threads>>>(output_tmp,nerv_top->output_data,len);
			CUDA_CHECK;
		}
		if(nerv_top->type=='s'){
			int blocks=(len+g_threads-1)/g_threads;
			g_top_sigmoid_deriv<<<blocks,g_threads>>>(output_tmp,nerv_top->output_data,len);
			CUDA_CHECK;
		}
		if(nerv_top->type=='l')
			CUBT(cublasScopy(cublasHandle, len, output_tmp, 1,nerv_top->output_data,1));
	}
	void cacul_deriv(double param,float **input,float **target){
		top_pre_deriv(param,target);
		for(int i=layers_num-1;i>0;i--){
			nervs[i]->get_deriv(nervs[i-1]->output);		
			nervs[i]->get_sub_deriv(nervs[i-1]->output,nervs[i-1]->type);
		}
		nervs[0]->get_deriv(input);
		for(int i=0;i<layers_num;i++)
			nervs[i]->get_decay_deriv(param,train_set.decay_mod);
	}
	void cacul_nerv(float **input,float **target) {	
		result=run(input,target);
		real_result=result;	
		for(int i=0;i<layers_num;i++)
			result+=train_set.decay*nervs[i]->weight_sum(train_set.decay_mod);
		cacul_deriv(train_set.decay,input,target);
	}

#else
	void init_struct(){	
		if(weight!=NULL){
			delete [] weight;
		}
		weight_len=0;
		for(int i=0;i<layers_num;i++){
			nervs[i]->init_struct();
			weight_len+=nervs[i]->weight_len;
			
		}
		weight=new float[weight_len];
		int p=0;
		for(int i=0;i<layers_num;i++){
			nervs[i]->init_weight(weight+p);
			p+=nervs[i]->weight_len;
		}		
	}
	void set_data_num(int dt_num){
		if(dt_num==data_num)return;
		data_num=dt_num;
		for(int i=0;i<layers_num;i++){
			nervs[i]->set_data_num(data_num);			
		}
	}
	void run(float *input,float *out){//调用此函数前必须先执行set_data_num指定数据数量
		int num=data_num;
		float **input_p=new float *[num];
		for(int i=0;i<num;i++)
			input_p[i]=input+i*input_dimen;
		nerv_bottom->run(input_p,num);
		for(int i=1;i<layers_num;i++){
			nervs[i]->run(nervs[i-1]->output,num);
		}
		memcpy(out,nerv_top->output_data,sizeof(float)*output_dimen*num);
		delete [] input_p;
	}
#endif	

};
}