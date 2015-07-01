class bagging{
public:
	ML_CLASS *main_nerv;
	ML_CLASS *nervs[1000];
	DATA_CLASS *dt_set;
	string root_path;
	int nervs_num;
	int input_dimen;
	int output_dimen;
	char vote_mod;
	bagging(string path,DATA_CLASS *data_set,bool trn=true){
		root_path=path;
		nervs_num=0;
		dt_set=data_set;
		vote_mod='a';//a:averge m:middle v:vote
		main_nerv=new ML_CLASS(path,dt_set,'r');		
		input_dimen=main_nerv->input_dimen;
		output_dimen=main_nerv->output_dimen;
		load_nervs(trn);
		if(trn) train();
	}
	string get_idx_name(){
		int idx=nervs_num+1;
		string path;
		char s[100];
		do{
			sprintf(s,"%i",idx);
			path=root_path+"bagging_"+s+"\\";
			_finddata_t FileInfo;
			long Handle = _findfirst(path.c_str(), &FileInfo); 
			
			if (Handle == -1L){
				_findclose(Handle);
				break;
			}
			idx++;
		}while(1);
		cout<<path;
		return path;
	}	
	void load_nervs(bool train){
		_finddata_t FileInfo;
		string search =root_path+"bagging_"+"*";
		long Handle = _findfirst(search.c_str(), &FileInfo); 
		if (Handle == -1L){
			cout<<endl<<"没有bagging项目\n";		
		} else   
		do{
			if (FileInfo.attrib &_A_SUBDIR){
				if( (strcmp(FileInfo.name,".") != 0 ) && (strcmp(FileInfo.name,"..") != 0)){	
						
						nervs_num++;						
				}
			}				
		}while (_findnext(Handle, &FileInfo) == 0); 
		_findclose(Handle);
		cout<<endl<<"发现"<<nervs_num<<"个bag";
	}
	void train(){
		while(1){
			coutd<<"************************";
			coutd<<"正在训练第"<<(nervs_num+1)<<"个bagging";
			string path=get_idx_name();
			if(!file_opt.check_folder(path))file_opt.create_folder(path);//return;
			if(!file_opt.copy(root_path+"params.stl",path+"params.stl"))return;
			if(!file_opt.copy(root_path+"pre_PCA.stl",path+"pre_PCA.stl"))return;
			if(!file_opt.copy(root_path+"struct.stl",path+"struct.stl"))return;
			if(nervs_num==0){nervs_num++;continue;}//第一个bagging直接从根目录中复制
			dt_set->bagging_set();
			ML_CLASS *nerv=new ML_CLASS(path,dt_set,'b');
			int num=nerv->ctr.total_rounds;
			nerv->train_reset();
			nerv->train(num);
			delete nerv;
			nervs_num++;
		}
	};
	string get_idx_name(int idx){
		_finddata_t FileInfo;
		string search =root_path+"bagging_"+"*";
		long Handle = _findfirst(search.c_str(), &FileInfo); 
		int i=0;
		do{
			if(i>=nervs_num)break;
			if (FileInfo.attrib &_A_SUBDIR){
				if(i==idx)return root_path+FileInfo.name+"\\";
				i++;
			
			}		
			
		}while (_findnext(Handle, &FileInfo) == 0); 
		_findclose(Handle);
	}
	float run(float *s_input,float *s_out,float *s_target,int num,float dropout=1.0f,bool in_cuda_pos=true,bool out_cuda_pos=true){
		float *input;
		float *out;
		float *target;
		if(in_cuda_pos){
			input=s_input;
		}else{
			cudaMalloc((void**)&input,sizeof(float)*input_dimen*num);
			CUDA_CHECK;
			cudaMemcpy(input,s_input,sizeof(float)*input_dimen*num,cudaMemcpyHostToDevice);
			CUDA_CHECK;
		}
		if(out_cuda_pos){
			out=new float[output_dimen*num];
			target=s_target;
			
		}else{
			out=s_out;
			cudaMalloc((void**)&target,sizeof(float)*output_dimen*num);
			CUDA_CHECK;
			cudaMemcpy(target,s_target,sizeof(float)*output_dimen*num,cudaMemcpyHostToDevice);
			CUDA_CHECK;
		
		}
		float *t_out;
		cudaMalloc((void**)&t_out,sizeof(float)*output_dimen*num);
		CUDA_CHECK;
		float *tt_out=new float[output_dimen*num];
		double *d_out=new double[output_dimen*num];
		float *sort_v=new float[output_dimen*num*nervs_num];
		float *v_out=new float[output_dimen*num];
		memzero(d_out);
		memzero(v_out);
		_finddata_t FileInfo;
		string search =root_path+"bagging_"+"*";
		long Handle = _findfirst(search.c_str(), &FileInfo); 
		int i=0;
		cout_show=false;
		do{
			if(i>=nervs_num)break;
			if (FileInfo.attrib &_A_SUBDIR){
				if( (strcmp(FileInfo.name,".") != 0 ) && (strcmp(FileInfo.name,"..") != 0)){	
				ML_CLASS nv(root_path+FileInfo.name+"\\",dt_set,'r');
				nv.run(input,t_out,target,num,1.0f/*nv.dropout_scl*/);
				cudaMemcpy(tt_out,t_out,sizeof(float)*output_dimen*num,cudaMemcpyDeviceToHost);		
				for(int j=0;j<num;j++){
					for(int k=0;k<output_dimen;k++){
						d_out[output_dimen*j+k]+=tt_out[output_dimen*j+k];
						sort_v[nervs_num*output_dimen*j+nervs_num*k+i]=tt_out[output_dimen*j+k];
						v_out[output_dimen*j+k]+=(tt_out[output_dimen*j+k]>0.5)?1:0;
					}
				}
				i++;
				}
			}		
			
		}while (_findnext(Handle, &FileInfo) == 0); 
		_findclose(Handle);		
		cout_show=true;
		for(int j=0;j<num;j++){
				for(int k=0;k<output_dimen;k++){
					if(vote_mod=='a')out[output_dimen*j+k]=d_out[output_dimen*j+k]/nervs_num;
					if(vote_mod=='m'){
						sort(sort_v+nervs_num*output_dimen*j+nervs_num*k,sort_v+nervs_num*output_dimen*j+nervs_num*k+nervs_num-1);
						out[output_dimen*j+k]=sort_v[nervs_num*output_dimen*j+nervs_num*k+nervs_num/2];
					}
					if(vote_mod=='v')out[output_dimen*j+k]=v_out[output_dimen*j+k]/nervs_num+d_out[output_dimen*j+k]/nervs_num/nervs_num/2;
				}
			}
		int dimen=output_dimen*num;
	int blocks=(dimen+g_threads-1)/g_threads;
	float *ov;
	cudaMalloc((void**)&ov,sizeof(float)*output_dimen*num);
	cudaMemcpy(t_out,out,sizeof(float)*output_dimen*num,cudaMemcpyHostToDevice);		
	gpu_loss_value<<<blocks,g_threads>>>(t_out,target,ov,dimen,main_nerv->loss_mod,output_dimen,dropout);	
	CUDA_CHECK;
	float ret=array_sum(ov,dimen)/num;
	CUBT(cublasSscal(cublasHandle,dimen, &dropout,t_out, 1));	
	cudaFree(ov);
	if(out_cuda_pos){
		cudaMemcpy(s_out,out,sizeof(float)*output_dimen*num,cudaMemcpyHostToDevice);
		CUDA_CHECK;
		delete [] out;
	}
	return ret;
	}
	
};