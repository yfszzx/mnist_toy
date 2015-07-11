
class load_train_data_class:public PCA{
public:
	struct data_pre_params{
		float noise_scale;
		int tmp;
		int compare_num;
		void show(){
			coutd<<"<数据预处理参数>";
			coutd<<"\t1.高斯噪音尺度"<<noise_scale;
			coutd<<"\t2.对照数据数量"<<compare_num;
		}
		void set(){
			char sel;
			show();
			coutd<<"\tc.确定 l.查看";
			while(1){
				coutd<<"预处理参数>";
				cin>>sel;
				switch(sel){
				case '1':
					cin>>noise_scale;
					break;			
				case '2':
					cin>>compare_num;
					break;
				case 'l':
					show();
					break;
				case 'c':
					return;
			};
		}
	}
	data_pre_params(){
			noise_scale=0;
			compare_num=10000;
		}
	};
	data_pre_params pre_params;
	int input_dimen;
	int output_dimen;
	int pre_dimen;
	float * input,*target,*pre_input;
	float * get_input,*get_target;
	//int	all_group_num;
	//int all_group_idx;
	//float *all_input,*all_target;
	//bool all_bl;
	int data_num,last_data_num;
	DATA_CLASS *data_set;
	string root;
	load_train_data_class(void *p,string path):PCA(path){		
		root=path;
		input=NULL;
		target=NULL;
		get_input=NULL;
		get_target=NULL;
		//all_input=NULL;
		//all_target=NULL;
		//all_bl=false;
		//all_group_idx=0;
		data_set=(DATA_CLASS *)p;
		data_num=0;
		output_dimen=data_set->output_dimen;
		pre_dimen=data_set->input_dimen;
		if(!pre_read()){
			pca_main(data_set->input_dimen);
			pre_read();
		}
		if(!read_params()){
			pre_params.set();
			save_params();
		}
		input_dimen=set.pca_dmn;
	 }
	~load_train_data_class(){
		if(get_input!=NULL)delete [] get_input;
		if(get_target!=NULL)delete [] get_target;
		if(input!=NULL)cudaFree(input);
		if(target!=NULL)cudaFree(target);
		CUDA_CHECK;
	}
	bool read_params(){
		string path=root+"pre_params.stl";
		ifstream fin(path,ios::binary);
		if(!fin)return false;
		coutd<<"正在读取"<<path;
		fin.read((char *)&pre_params,sizeof(data_pre_params));
		fin.close();
		return true;
	}
	void save_params(){
		string path=root+"pre_params.stl";
		ofstream fin(path,ios::binary);
		coutd<<"正在保存"<<path;
		fin.write((char *)&pre_params,sizeof(data_pre_params));
		fin.close();
	}
	void pre_load(int dt_num,char mod='t'){		
		 last_data_num=data_num;
		 if(data_num!=dt_num){
			 data_num=dt_num;
			 safe_free(get_input);
			 safe_free(get_target);
			 get_input=new float[set.dimen*data_num];
			 get_target=new float[output_dimen*data_num];
		 }
		  data_set->get_data(mod,get_input,get_target,data_num);	
		  //memcpy(get_input,all_input+set.dimen*data_num*all_group_idx,sizeof(float)*set.dimen*data_num);
		  //memcpy(get_target,all_target+output_dimen*data_num*all_group_idx,sizeof(float)*output_dimen*data_num);

		/* if(all_group_idx==0){
			 safe_free(all_input);
			 safe_free(all_target);
			all_input=new float[set.dimen*data_num*all_group_num];
			all_target=new float[output_dimen*data_num*all_group_num];
		}
		// coutd<<all_group_idx<<" "<<all_group_num<<" "<<all_bl<<" "<<data_num<<" "<<set.dimen<<" "<<output_dimen;
		 //coutd<<_msize(all_target)<<" "<<_msize(all_input)<<" "<<_msize(get_input);
		 if(!all_bl&&mod=='t'){
			 data_set->get_data(mod,get_input,get_target,data_num,pre_params.distortion_scale);	
			 memcpy(all_input+set.dimen*data_num*all_group_idx,get_input,sizeof(float)*set.dimen*data_num);
			 memcpy(all_target+output_dimen*data_num*all_group_idx,get_target,sizeof(float)*output_dimen*data_num);
			if(all_group_idx==all_group_num-1)all_bl=true;
		 }else{
			 all_group_idx%=all_group_num;
			 memcpy(get_input,all_input+set.dimen*data_num*all_group_idx,sizeof(float)*set.dimen*data_num);
			 memcpy(get_target,all_target+output_dimen*data_num*all_group_idx,sizeof(float)*output_dimen*data_num);
		 }
		 all_group_idx++;*/
	 }
	void load_data(){
		if(last_data_num!=data_num){
			if(input!=NULL){
				cudaFree(input);
				cudaFree(pre_input);
			}
			if(target!=NULL)cudaFree(target);
			CUDA_CHECK;
			cudaMalloc((void**)&input,sizeof(float)*data_num*input_dimen);
			cudaMalloc((void**)&pre_input,sizeof(float)*data_num*set.dimen);
			cudaMalloc((void**)&target,sizeof(float)*data_num*output_dimen);
			CUDA_CHECK;
		}
		cudaMemcpy(pre_input,get_input,sizeof(float)*set.dimen*data_num,cudaMemcpyHostToDevice);
		data_set->gpu_distortion(pre_input,data_num);		
		trans_data(pre_input,input,data_num,pre_params.noise_scale);
		cudaMemcpy(target,get_target,sizeof(float)*output_dimen*data_num,cudaMemcpyHostToDevice);
		CUDA_CHECK;		
		
	}
	void get_pre_data(float *dt,int num){//获取预处理的数据
		pre_load(num,'p');
		cudaMemcpy(dt,get_input,sizeof(float)*set.dimen*num,cudaMemcpyHostToDevice);
		data_set->gpu_distortion(dt,num);
	}
	void reset_pca(){
		int dmn=PCA::set.dimen;
		pca_main(PCA::set.dimen,true);
		if(input_dimen!=PCA::set.pca_dmn){
			input_dimen=PCA::set.pca_dmn;
		}
	}
	virtual void cacual_compare(char mod='c')=0;
	void get_compare_data(char mod){		
		cmp_data &s=(mod=='c')?data_set->compare_data:data_set->train_cmp_data;
		if(s.num>0)return;
		data_set->get_compare_data(pre_params.compare_num,mod);
		cudaMalloc((void **)&(s.trans_input),sizeof(float)*input_dimen*pre_params.compare_num);
		trans_data(s.input,s.trans_input,pre_params.compare_num);
	}
	void get_compare_data(){
		get_compare_data('t');
		get_compare_data('c');
	}
	float show_compare(char mod='c'){
		if(pre_params.compare_num==0)return 0;		
		cmp_data &s=(mod=='c')?data_set->compare_data:data_set->train_cmp_data;		
		cacual_compare(mod);
		cudaMemcpy(s.cpu_output,s.output,sizeof(float)*output_dimen*pre_params.compare_num,cudaMemcpyDeviceToHost);	
		CUDA_CHECK;
		data_set->show_compare(mod);
		return s.result;		
	}	
};