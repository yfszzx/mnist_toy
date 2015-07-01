__global__ void gpu_loss_function(float *out,float *target,float *o,float *deriv,int dimen,char loss_mod,int output_dimen){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;	
	if(idx<dimen){
		if(loss_mod=='s'){//soft_max
			double sum;
			for(int i=0;i<output_dimen;i++){
				sum+=out[idx-idx%output_dimen+i];					
			}
			if(sum<0.000001)sum=0.000001;
			double y=out[idx]/sum;
			if(y<0.000001)y=0.000001;
			if(target[idx]>0){
				o[idx]=-__logf(y);
				deriv[idx]=-(1.0f-y)/y/sum;
			}
			else {
				o[idx]=0;
				deriv[idx]=1/sum;
			}
			return;
		}

		float ot=out[idx];	
		if(loss_mod=='c'){//互熵损失
			float tgt=target[idx];
			float y=ot;
			if(y<0.0001)y=0.0001;
			if(y>0.9999)y=0.9999;			
			o[idx]=-tgt*__logf(y)-(1-tgt)*__logf(1-y);
			deriv[idx]=-(tgt/y-(1-tgt)/(1-y));
			return;
		}		

		float dev=ot-target[idx];
		switch(loss_mod){
			case '2':
				o[idx]=dev*dev;
				deriv[idx]=2*dev;
				break;
			case '1':
				o[idx]=abs(dev);
				deriv[idx]=((dev>0)?1:-1);
				break;		
			case '3':
				o[idx]=abs(dev*dev*dev);
				deriv[idx]=((dev>0)?3*dev*dev:(-3)*dev*dev);
				break;

			}
		}

}
__global__ void gpu_loss_value(float *out,float *target,float *o,int dimen,char loss_mod,int output_dimen,float dropout_scl){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;	
	if(idx<dimen){
		if(loss_mod=='s'){//soft_max
			double sum;
			for(int i=0;i<output_dimen;i++){
				sum+=out[idx-idx%output_dimen+i];					
			}
			if(sum<0.000001)sum=0.000001;
			double y=out[idx]/sum*dropout_scl;
			if(y<0.000001)y=0.000001;
			if(target[idx]>0){
				o[idx]=-__logf(y);
				
			}
			else {
				o[idx]=0;
				
			}
			return;
		}

		float ot=out[idx]*dropout_scl;	
		if(loss_mod=='c'){//互熵损失
			float tgt=target[idx];
			float y=ot;
			if(y<0.0001)y=0.0001;
			if(y>0.9999)y=0.9999;			
			o[idx]=-tgt*__logf(y)-(1-tgt)*__logf(1-y);
			return;
		}		

		float dev=ot-target[idx];
		switch(loss_mod){
			case '2':
				o[idx]=dev*dev;
				break;
			case '1':
				o[idx]=abs(dev);
				break;		
			case '3':
				o[idx]=abs(dev*dev*dev);
				break;

			}
		}

}
class multi_perceptrons{
public:
	int input_dimen;
	int output_dimen;
	int layers_num;
	int weight_len;
	float *weight;	
	float *deriv;
	float result;
	float real_result;
	bool train_mod;
	char loss_mod;	
	bool reguler_mod;
	bool save_history;
	perceptrons *nerv_bottom,*nerv_top,**nervs;
	int data_num;
	string path;
	string proj_path;
	
	multi_perceptrons(string name){
		nervs=NULL;
		output_tmp=NULL;
		layers_num=0;
		weight_len=0;	
		weight=NULL;
		deriv=NULL;
		srand((int)time(NULL));
		data_num=0;
		train_mod=false;
		proj_path=name;
		path=name+"struct.stl";
		loss_mod='2';
		reguler_mod=0;
		save_history=false;
	}
		void init_nervs(){			
		if(nervs!=NULL){
			for(int i=0;i<layers_num;i++){
				delete nervs[i];
			}
			delete [] nervs;	
		}
		nervs=new perceptrons *[layers_num];		
		for(int i=0;i<layers_num;i++)nervs[i]=new perceptrons();		
		nerv_bottom=nervs[0];
		nerv_top=nervs[layers_num-1];
	}
	void struct_init(){	
		safe_gpu_free(weight);
		safe_gpu_free(deriv);
		weight_len=0;
		for(int i=0;i<layers_num;i++){
			nervs[i]->init_struct();
			weight_len+=nervs[i]->weight_len;
		}
		cudaMalloc((void**)&weight,sizeof(float)*weight_len);	
		CUDA_CHECK;		
		if(train_mod){
			cudaMalloc((void**)&deriv,sizeof(float)*weight_len);
			CUDA_CHECK;
		}
		int p=0;
		for(int i=0;i<layers_num;i++){
			nervs[i]->weight=weight+p;
			if(train_mod)nervs[i]->deriv=deriv+p;
			p+=nervs[i]->weight_len;
		}	
		input_dimen=nerv_bottom->input_dimen;
		output_dimen=nerv_top->nodes_num;
	}
	void struct_settted(bool save=true){
		struct_init();
		weight_rand();
		struct_show();
		if(save)struct_save();
	}
	void weight_rand(){
		for(int i=0;i<layers_num;i++)
			nervs[i]->weight_rand();
	}
	void set_data_num(int dt_num){
		if(dt_num==data_num)return;
		 free_mem();
		data_num=dt_num;
		for(int i=0;i<layers_num;i++){
			nervs[i]->set_data_num(data_num);
		}
		cudaMalloc((void**)&output_tmp,sizeof(float)*output_dimen*data_num);
		cudaMalloc((void**)&tmp_array,sizeof(float)*output_dimen*data_num);
	}
	float get_result(float *input,float *target){			
		nerv_bottom->run(input);
		for(int i=1;i<layers_num;i++){
			nervs[i]->run(nervs[i-1]->output);
		}
		int dimen=output_dimen*data_num;
		CUDA_CHECK;
		int blocks=(dimen+g_threads-1)/g_threads;
		gpu_loss_function<<<blocks,g_threads>>>(nerv_top->output,target,tmp_array,output_tmp,dimen,loss_mod,output_dimen);	
		CUDA_CHECK;
		float ret=array_sum(tmp_array,dimen)/data_num;
		CUDA_CHECK;
		return ret;	
	}
	float run(float *s_input,float *out,float *s_target,int num,float dropout=1.0f,int out_layer=-1,bool in_cuda_pos=true,bool out_cuda_pos=true){
		set_data_num(num);
		if(out_layer==-1)out_layer=layers_num;
		float *input;
		float *target;
		if(in_cuda_pos){
			input=s_input;
		}else{
			cudaMalloc((void**)&input,sizeof(float)*input_dimen*data_num);
			CUDA_CHECK;
			cudaMemcpy(input,s_input,sizeof(float)*input_dimen*data_num,cudaMemcpyHostToDevice);
			CUDA_CHECK;

		}
		if(out_cuda_pos){
			target=s_target;
		}else{
			cudaMalloc((void**)&target,sizeof(float)*output_dimen*data_num);
			CUDA_CHECK;
			cudaMemcpy(target,s_target,sizeof(float)*output_dimen*data_num,cudaMemcpyHostToDevice);
			CUDA_CHECK;			
		}
		nerv_bottom->run(input);
		for(int i=1;i<out_layer;i++){
			nervs[i]->run(nervs[i-1]->output);
		}
		int dimen=nervs[out_layer-1]->nodes_num*num;
		int blocks=(dimen+g_threads-1)/g_threads;
		float *ov;
		cudaMalloc((void**)&ov,sizeof(float)*output_dimen*num);
		gpu_loss_value<<<blocks,g_threads>>>(nervs[out_layer-1]->output,target,ov,dimen,loss_mod,output_dimen,dropout);		
		CUDA_CHECK;
		float ret=array_sum(ov,dimen)/data_num;		
		CUBT(cublasSscal(cublasHandle,dimen, &dropout,nervs[out_layer-1]->output, 1));	
		cudaMemcpy(out,nervs[out_layer-1]->output,sizeof(float)*dimen,((out_cuda_pos)?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));		
		CUDA_CHECK;
		cudaFree(ov);
		if(!in_cuda_pos){
			cudaFree(input);
			CUDA_CHECK;
		}
		if(!out_cuda_pos){			
			cudaFree(target);
		}
		return ret;
	}
	void top_pre_deriv(){
		int len=output_dimen*data_num;
		if(nerv_top->type=='t'){
			//int blocks=(len+g_threads-1)/g_threads;
			g_top_tanh_deriv/*<<<blocks,g_threads>>>*/(output_tmp,nerv_top->output,len);
			
		}
		if(nerv_top->type=='s'){
		//	int blocks=(len+g_threads-1)/g_threads;
			g_top_sigmoid_deriv/*<<<blocks,g_threads>>>*/(output_tmp,nerv_top->output,len);

		}
		if(nerv_top->type=='l')
			CUBT(cublasScopy(cublasHandle, len, output_tmp, 1,nerv_top->output,1));
	}
	void cacul_deriv(float *input,float *target){
		top_pre_deriv();
		for(int i=layers_num-1;i>0;i--){
			nervs[i]->get_deriv(nervs[i-1]->output);		
			nervs[i]->get_sub_deriv(nervs[i-1]->output,nervs[i-1]->type);
		}
		nervs[0]->get_deriv(input);
		if(reguler_mod){
			for(int i=0;i<layers_num;i++)
				nervs[i]->get_decay_deriv();
		}
	}
	void cacul_nerv(float *input,float *target) {	
		result=get_result(input,target);
		real_result=result;	
		if(reguler_mod){
			for(int i=0;i<layers_num;i++)
				result+=nervs[i]->weight_decay_sum();
		}
		cacul_deriv(input,target);
		
	}
	void struct_show(){

		if(layers_num==0)cout<<"还未初始化网络结构\n";
		else {
			coutd<<"\t\t\t<    MLP结构    >\n";
			coutd<<"输入:"<<input_dimen<<"维  输出:"<<output_dimen<<"维 隐层数:"<<layers_num-1<<" 层"<<" 参数数量:"<<weight_len<<" 样本数:"<<data_num;
			coutd;
			coutd<<"\t(激励函数类型:s sigmoid, t tanh,l 线性函数 )\n";
			for(int i=0;i<layers_num;i++)
				nervs[i]->show_struct(i+1);
			coutd;
			coutd<<"损失函数(1.L1 2.L2 c.互熵 s.soft_max):"<<loss_mod;
			coutd;			
			if(reguler_mod) reguler_show();
			coutd;
			memery();
		}

	}	
	bool struct_read(){			
		ifstream fin(path,ios::binary);
		if(!fin)return false;
		coutd<<"正在读取结构文件"<<path;
		fin.read((char *)&layers_num,sizeof(int));
		fin.read((char *)&loss_mod,sizeof(char));
		fin.read((char *)&reguler_mod,sizeof(bool));
		init_nervs();
		for(int i=0;i<layers_num;i++)nervs[i]->read_struct(&fin);
		
		struct_init();	
		
		struct_show();
		float *tmp=new float[weight_len];	
		fin.read((char *)tmp,sizeof(float)*weight_len);
		cudaMemcpy(weight,tmp,sizeof(float)*weight_len,cudaMemcpyHostToDevice);
		CUDA_CHECK;
		
		fin.close();	
		delete [] tmp;
	
		return true;
		
	}
	void struct_save(int idx=-1){
		coutd<<"正在保存"<<path;
		ofstream fin(path,ios::binary);
		fin.write((char *)&layers_num,sizeof(int));
		fin.write((char *)&loss_mod,sizeof(char));
		fin.write((char *)&reguler_mod,sizeof(bool));
		for(int i=0;i<layers_num;i++)nervs[i]->write_struct(&fin);
		float *tmp=new float[weight_len];
		cudaMemcpy(tmp,weight,sizeof(float)*weight_len,cudaMemcpyDeviceToHost);
		CUDA_CHECK;
		fin.write((char *)tmp,sizeof(float)*weight_len);
		delete [] tmp;	
		fin.close();	
		if(idx>0&&save_history){
			stringstream s;
			s<<proj_path<<"struct"<<idx<<".stl";
			file_opt.copy(path,s.str());
		}
		

	}
	void reguler_set(){
		if(reguler_mod==0){
			coutd<<"是否使用正则化?(Y/N)";
			reguler_mod=0;		
			string sel;
			cin>>sel;
			if(sel[0]=='y'||sel[0]=='Y'){
				reguler_mod=1;
			}else return;
		}
		reguler_show();
		do{ 
			int idx=select_layer("(输入0,不使用正则模式;输入-1退出;输入-2,查看正则模式)");
			if(idx==0){
				reguler_mod=0;
				break;
			
			}
			if(idx==-2){
				reguler_show();
				continue;
			}
			if(idx==-1)break;
			coutd<<"输入正则模式及正则系数(空格分隔)>";
			cin>>nervs[idx-1]->decay_mod>>nervs[idx-1]->decay;
		}while(1);

	}
	void loss_mod_set(){
		coutd<<"输入损失函数类型(1.L1 2.L2 c.互熵 s.soft_max):";
		cin>>loss_mod;
	}
	void reguler_show(){
		coutd<<"各层正则模式（1:L1 2:L2 ) 和正则系数:";
		for(int i=0;i<layers_num;i++)
			coutd<<"\t"<<(i+1)<<" mod:"<<nervs[i]->decay_mod<<" decay:"<<nervs[i]->decay;
		
	}
	void struct_set(int i_dimen=0,int o_dimen=0){
		if(i_dimen)input_dimen=i_dimen;
		if(o_dimen)output_dimen=o_dimen;
		coutd<<"\t<设置网络结构>";
		coutd<<"输入隐层数量:";
		if(layers_num>0)cout<<"(当前隐层数"<<layers_num-1<<"):";
		cin>>layers_num;
		layers_num++;
		init_nervs();
		struct_show();	
		coutd<<"输入各个隐层的神经节数量、激励函数类型(空格分隔):";	
		nervs[0]->input_dimen=input_dimen;
		nervs[layers_num-1]->nodes_num=output_dimen;
		for(int i=0;i<layers_num-1;i++){
			coutd<<"第"<<i+1<<"层:";
			cin>>nervs[i]->nodes_num>>nervs[i]->type;
			nervs[i+1]->input_dimen=nervs[i]->nodes_num;
		}
		coutd<<"设置输出层激励函数类型:";
		cin>>nervs[layers_num-1]->type;
		struct_settted();
		loss_mod_set();
		coutd<<"是否使用正则化?(Y/N)";
		reguler_mod=0;		
		string sel;
		cin>>sel;
		if(sel[0]=='y'||sel[0]=='Y'){
			reguler_mod=1;
			reguler_set();
		}
	
	}	
	void struct_simple_set(int in_dimen,int out_dimen,int nodes,char nodes_mod,char out_mod,char loss='2',float  decay=0,char decay_mod='2'){
		if(decay==0)reguler_mod=0;
		else reguler_mod=1;
		input_dimen=in_dimen;
		output_dimen=out_dimen;
		layers_num=2;
		init_nervs();
		nervs[0]->input_dimen=input_dimen;
		nervs[0]->nodes_num=nodes;
		nervs[0]->type=nodes_mod;
		nervs[0]->decay_mod=decay_mod;
		nervs[0]->decay=decay;
		nervs[1]->input_dimen=nodes;
		nervs[1]->nodes_num=out_dimen;
		nervs[1]->type=out_mod;
		nervs[1]->decay_mod=decay_mod;
		nervs[1]->decay=decay;
		struct_settted(false);
		loss_mod=loss;
	}
	int select_layer(string illu=""){
		int idx;
		do{		
			coutd<<"输入要修改的层序:";
			coutd<<illu;
			coutd<<"输入>";
			cin>>idx;
			if(idx>layers_num){
				coutd<<"【错误】";
				continue;
			}
			break;
		}while(1);
		return idx;
	}
	void struct_edit(){		
		struct_show();
		string sel;
		coutd<<"\t1.修改损失函数";
		coutd<<"\t2.修改正则模式";
		coutd<<"\t3.修改激励函数";
		coutd<<"\t4.修改结点数量";
		coutd<<"\t5.重置网络结构";
		coutd<<"\tl.查看";
		coutd<<"\tc.确定";
		while(1){
			coutd<<"修改MLP设置>";
			cin>>sel;
			switch(sel[0]){
			case '1':
				loss_mod_set();
				break;
			case '2':
				reguler_set();
				break;
			case '3':
				do{
				int idx=select_layer("(输入-1结束)");
				if(idx==-1)break;
				coutd<<"输入激励函数类型:";
				cin>>nervs[idx-1]->type;
				}while(1);
				break;
			case '4':
				do{
				int idx=select_layer("(输入-1结束)");
				if(idx==-1)break;
				if(idx==layers_num){
					coutd<<"【提示】不能修改顶层节点数量";
					continue;
				}
				coutd<<"输入节点数量:";
				cin>>nervs[idx-1]->nodes_num;
				nervs[idx]->input_dimen=nervs[idx-1]->nodes_num;
				}while(1);
				struct_settted();
				break;
			case '5':
				struct_set();
				break;
			case 'l':
				struct_show();
				break;
			case 'c':
				return;
			}
		}
	}

	~multi_perceptrons(){
		if(nervs!=NULL){
			for(int i=0;i<layers_num;i++)
				delete nervs[i];
			delete [] nervs;
		}
		safe_gpu_free(weight);
		safe_gpu_free(deriv);
		free_mem();
	}
	int memery(bool show=true){
		int ret=sizeof(float)*weight_len*2+sizeof(float)*output_dimen*data_num*2;
		for(int i=0;i<layers_num;i++)
				ret+=nervs[i]->memery();		
		if(show)show_memery_size(ret,'g',"总计占用");
		return ret;
	}
	void reset(){
		weight_rand();
		struct_save();
		
	}
	
	
	private:
	float *output_tmp;
	float *tmp_array;
	void free_mem(){
		if(output_tmp!=NULL){
			cudaFree(output_tmp);
			cudaFree(tmp_array);
		};
	}
};
class multi_perceptrons_sample:public search_tool{
public:
	float *input;
	float *output;
	multi_perceptrons *mlp;
	multi_perceptrons_sample(string path){
		g_gpu_init();
		mlp=new multi_perceptrons(path);
		mlp->train_mod=true;
		int num;
		char np,op;
		coutd<<"分别输入神经节数量、隐层和输出层激励函数类型，空格分格：";
		coutd<<"[激励函数类型  l:线性，s:sigmoid，t:tanh]";
		cin>>num>>np>>op;
		mlp->struct_simple_set(1,1,num,np,op);
		int data_num;
		coutd<<"输入训练数据数量:";
		cin>>data_num;
		mlp->set_data_num(data_num);
		set.debug=true;		
		search_init(mlp->weight_len,&(mlp->result),mlp->weight,mlp->deriv);
		set.set();
		set_init();
		srand((int)time(0));
		cudaMalloc((void**)&input,sizeof(float)*data_num);
		cudaMalloc((void**)&output,sizeof(float)*data_num);
		float *t1=new float[data_num];
		float *t2=new float[data_num];
		for(int i=0;i<data_num;i++){
			t1[i]=float(rand())/RAND_MAX*(rand()%2*2-1)*3.1415;
			t2[i]=sin(t1[i]);
		}
		cudaMemcpy(input,t1,sizeof(float)*data_num,cudaMemcpyHostToDevice);
		CUDA_CHECK;
		cudaMemcpy(output,t2,sizeof(float)*data_num,cudaMemcpyHostToDevice);
		CUDA_CHECK;
		delete [] t1;
		delete [] t2;
		for(int i=0;i<1000;i++){
		search(10000);
		}
		string s;
		cin>>s;
	}
	virtual bool show_and_control(int i){
		coutd<<i<<" "<<*result;
		return true;
	}
	virtual bool pause_action(){
		getchar();
		return true;
	}
	virtual void cacul(){
		mlp->cacul_nerv(input,output);
	}
};
/*
class sample:public virtual_data_set{
public:
	sample(){
		input_dimen=2;
		output_dimen=2;
		action_illu="未定义";
	}
	void get_data(char mod,float *input,float *target,int data_num,float distortion_scale){
				for(int i=0;i<data_num;i++){
					input[i*2]=float(rand())/RAND_MAX*(rand()%2*2-1)*3.1415;
					input[i*2+1]=float(rand())/RAND_MAX*(rand()%2*2-1)*3.1415;
					target[i*2]=sin(input[i*2]);
					target[i*2+1]=sin(input[i*2+1]);
				}
		
	}
	virtual void show_compare(){
		coutd<<"sample show compare";
	};
	virtual void self_action(){};
};
*/
