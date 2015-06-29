
class load_train_data_class:public PCA{
public:
	struct data_pre_params{
		float noise_scale;
		float distortion_scale;
		int show_freq;
		int compare_num;
		void show(){
			coutd<<"<数据预处理参数";
			coutd<<"\t1.噪音程度"<<noise_scale;
			coutd<<"\t2.扭曲程度"<<distortion_scale;
			coutd<<"\t3.对照数据数量"<<compare_num;
			coutd<<"\t4.对照数据显示间隔"<<show_freq;
		}
		void set(){
			char sel;
			show();
			coutd<<"\tc.确定 l.查看";
			while(1){
				coutd<<"设置>";
				cin>>sel;
				switch(sel){
				case '1':
					cin>>noise_scale;
					break;			
				case '2':
					cin>>distortion_scale;
					break;
				case '3':
					cin>>compare_num;
					break;
				case '4':
					cin>>show_freq;
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
			distortion_scale=0;
			show_freq=0;
			compare_num=0;
		}
	};
	data_pre_params pre_params;
	int input_dimen;
	int output_dimen;
	int pre_dimen;
	float * input,*target,*pre_input;
	float * get_input,*get_target;
	int	all_group_num;
	int all_group_idx;
	float *all_input,*all_target;
	bool all_bl;
	int data_num,last_data_num;
	DATA_CLASS *data_set;
	load_train_data_class(DATA_CLASS *p,string path):PCA(path){		
		input=NULL;
		target=NULL;
		get_input=NULL;
		get_target=NULL;
		all_input=NULL;
		all_target=NULL;
		all_bl=false;
		all_group_idx=0;
		data_set=p;
		data_num=0;
		output_dimen=p->output_dimen;
		pre_dimen=p->input_dimen;
		if(!pre_read()){
			pca_main(data_set->input_dimen);
			pre_read();
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
	void pre_load(int dt_num,char mod='t'){	
	
		last_data_num=data_num;
		 if(data_num!=dt_num){
			 data_num=dt_num;
			 if(get_input!=NULL)delete [] get_input;
			 if(get_target!=NULL)delete [] get_target;
			 get_input=new float[set.dimen*data_num];
			 get_target=new float[output_dimen*data_num];
		 }

		 if(all_group_idx==0){
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
		 all_group_idx++;
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
			CUDA_CHECK;
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
	virtual void cacual_compare(char mod='c')=0;
	void show_compare(char mod='c'){
		if(pre_params.compare_num==0)return;		
		virtual_data_set::cmp_data &s=(mod=='c')?data_set->compare_data:data_set->train_cmp_data;
		if(s.num==0){
				data_set->get_compare_data(pre_params.compare_num,mod);
				cudaMalloc((void **)&(s.trans_input),sizeof(float)*input_dimen*pre_params.compare_num);
				trans_data(s.input,s.trans_input,pre_params.compare_num);			
		}
		cacual_compare(mod);
		cudaMemcpy(s.cpu_output,s.output,sizeof(float)*output_dimen*pre_params.compare_num,cudaMemcpyDeviceToHost);	
		CUDA_CHECK;
		data_set->show_compare(mod);
	}
};
void *pthread_train(void *);
void *pthread_preload(void *);
class multi_perceptrons_dropout:public multi_perceptrons{
public:
	multi_perceptrons *main;
	float dropout_scl;
	int dropout_num;
	int real_num;
	int *dropout_list;
	float *real_weight;
	multi_perceptrons_dropout(string name,bool train=false):multi_perceptrons(name,train){
		main=new multi_perceptrons(name,false);		
		dropout_scl=0.5;
		train_mod=train;
	}
	void make_struct(){
		real_num=main->nervs[0]->nodes_num;
		dropout_num=real_num*dropout_scl;
			
		struct_simple_set(main->input_dimen,main->output_dimen,dropout_num,main->nervs[0]->type,main->nervs[1]->type);
		real_weight=new float[main->weight_len];
		dropout_list=new int[weight_len];	
		cudaMemcpy(real_weight,main->weight,sizeof(float)*main->weight_len,cudaMemcpyDeviceToHost);
		dropout_trans_weight();
	}

	bool struct_read(){
		bool ret=main->struct_read();
		if(ret)make_struct();
		return ret;
	}
	void struct_set(int in_dimen,int out_dimen){
		main->struct_set(in_dimen,out_dimen);
		make_struct();
	}
	void struct_edit(){
		main->struct_edit();
		make_struct();
	}
	void set_data_num(int num){
		main->set_data_num(num);
		multi_perceptrons::set_data_num(num);

	}
	void struct_save(){
		dropout_trans_back();
		cudaMemcpy(main->weight,real_weight,sizeof(float)*main->weight_len,cudaMemcpyHostToDevice);
		main->struct_save();
	}
	void dropout_trans_weight(){
		int idx;
		float *wd;
		wd=new float[weight_len];
		float *topd=wd+nervs[0]->weight_len;
		float *topr=real_weight+main->nervs[0]->weight_len;	
		int pos=0;
		for(int i=0;i<dropout_num;i++){
			//rand()%real_num;//(rand()%2*real_num)/2+i;
			
			for(int j=0;j<input_dimen;j++){
				idx=(rand()%2*dropout_num)+i;	
				dropout_list[pos]=idx;
				pos++;
				wd[dropout_num*j+i]=real_weight[real_num*j+idx];
				
			}
			idx=(rand()%2*dropout_num)+i;	
			dropout_list[pos]=idx;
			pos++;
			wd[input_dimen*dropout_num+i]=real_weight[input_dimen*real_num+idx];
			for(int j=0;j<output_dimen;j++){
				idx=(rand()%2*dropout_num)+i;	
				dropout_list[pos]=idx;
				pos++;
				topd[output_dimen*i+j]=topr[output_dimen*idx+j];
			}
			
		}
		for(int j=0;j<output_dimen;j++){
				
				topd[dropout_num*output_dimen+j]=topr[real_num*output_dimen+j]*dropout_scl;						
		}
		cudaMemcpy(weight,wd,sizeof(float)*weight_len,cudaMemcpyHostToDevice);
		CUDA_CHECK;
		delete [] wd;
	}

	void dropout_trans_back(){
		int idx;
		float *wd;
		wd=new float[weight_len];
		cudaMemcpy(wd,weight,sizeof(float)*weight_len,cudaMemcpyDeviceToHost);
		CUDA_CHECK;
		float *topd=wd+nervs[0]->weight_len;
		float *topr=real_weight+main->nervs[0]->weight_len;		
		int pos=0;
		for(int i=0;i<dropout_num;i++){
				
			for(int j=0;j<input_dimen;j++){
				idx=dropout_list[pos];	
				pos++;
				real_weight[real_num*j+idx]=wd[dropout_num*j+i];
			}
			idx=dropout_list[pos];	
			pos++;
			real_weight[input_dimen*real_num+idx]=wd[input_dimen*dropout_num+i];
			for(int j=0;j<output_dimen;j++){
				idx=dropout_list[pos];	
				pos++;
				topr[output_dimen*idx+j]=topd[output_dimen*i+j];
			}
			
		}
		for(int j=0;j<output_dimen;j++){
				topr[real_num*output_dimen+j]=topr[real_num*output_dimen+j]*(1-dropout_scl)+topd[dropout_num*output_dimen+j];						
		}
	
		delete [] wd;
	}
};
class multi_perceptrons_train: public multi_perceptrons,public search_tool,public load_train_data_class{
public:
	string root;
	int seeder;
	int input_dimen;
	int output_dimen;
	float init_result;
	void self_action();
	void check_result();
	void report();
	multi_perceptrons_train(string name,DATA_CLASS *dt,char mod='t'):multi_perceptrons(name,true),load_train_data_class(dt,name){
		//mod：t train r read b bagging c check
		g_gpu_init();
		root=name;
		input_dimen=load_train_data_class::input_dimen;
		output_dimen=load_train_data_class::output_dimen;
		if(!struct_read())struct_set(input_dimen,output_dimen);
		if(!read_params()){
			ctr.set();
			set_search();
			pre_params.set();
			save_params();
		}	
		if(mod=='r')return;//read_only
		if(mod=='c'){//check;
			self_action();
			return;
		}
		search_init(weight_len,&(multi_perceptrons::result),weight,multi_perceptrons::deriv);
		ctr.init();	
		if(mod=='d')return;//dropout;
		if(mod=='b')reset();//bagging;
		if(mod=='t')if(!operate())return;//train;
		train();
	
		
	}
	bool read_params(){
		string path=root+"params.stl";
		ifstream fin(path,ios::binary);
		if(!fin)return false;
		coutd<<"正在读取"<<path;
		fin.read((char *)&(search_tool::set),sizeof(search_set));
		fin.read((char *)&ctr,sizeof(controller)-sizeof(float *));
		fin.read((char *)&pre_params,sizeof(data_pre_params));
		fin.close();
		return true;
	}
	void save_params(){
		string path=root+"params.stl";
		coutd<<"正在保存"<<path;
		ofstream fin(path,ios::binary);
		fin.write((char *)&(search_tool::set),sizeof(search_set));
		fin.write((char *)&ctr,(sizeof(controller)-sizeof(float *)));
		fin.write((char *)&pre_params,sizeof(data_pre_params));
		fin.close();
	}

	void menu(){
		coutd;
		coutd<<"\t1.设置网络";
		coutd<<"\t2.重新初始化";
		coutd<<"\t3.设置训练参数";
		coutd<<"\t4.设置搜索参数";
		coutd<<"\t5.设置预处理参数";
		coutd<<"\t6.保存结果";
		coutd<<"\t7."<<data_set->action_illu;
		coutd<<"\t8."<<"重新预处理数据";
		
		coutd<<"\tc.开始(继续)训练 l.菜单 e.退出";
	}
	void reset_pca(){
		int dmn=PCA::set.dimen;
		pca_main(PCA::set.dimen,true);
		if(input_dimen!=PCA::set.pca_dmn){
			load_train_data_class::input_dimen=PCA::set.pca_dmn;
			input_dimen=load_train_data_class::input_dimen;
			struct_set(input_dimen,output_dimen);
		}
	}

	bool operate(){
		menu();
		while(1){
			char s;
			coutd<<"选择>";
			cin>>s;
			switch(s){
			case '1':
				struct_edit();
				break;
			case '2':
				reset();
				break;
			case '3':
				ctr.set();
				save_params();
				break;
			case '4':
				set_search();
				save_params();
				break;
			case '5':
				pre_params.set();
				save_params();
				break;
			case '6':
				struct_save();
				save_params();
				break;
			case '7':
				self_action();
				break;
			case '8':
				 reset_pca();
				break;
			case 'c':
				return true;
			case 'e':
				return false;
			case 'l':
				menu();
			}
		}
	}
	void cacul_parallel_load(){
			seeder=rand();
			pthread_t tid1,tid2;
			pthread_create(&tid1,NULL,&(pthread_train),this);
			pthread_create(&tid2,NULL,&(pthread_preload),this);
			void *ret1,*ret2;
			pthread_join(tid1,&ret1);	
			set_data_num(ctr.data_num);
			pthread_join(tid2,&ret2);
			//dropout_trans_back();
			//dropout_trans_weight();
			load_data();
	}
	void train_bigenning(){
		all_group_num=ctr.rounds_init;
		pre_load(ctr.data_num);
		load_data();
		set_data_num(ctr.data_num);
		show_compare();
		show_compare('t');
		//	dropout_trans_weight();
	}
	void train(){
			ctr.real_rounds=5;
			ctr.rounds_init=1;
		train_bigenning();
		int cnt=0;
		
		while(1){
			cnt++;
				ctr.real_rounds=5;
			ctr.rounds_init=1;
			search_tool::search(ctr.real_rounds);
			/*if(!ctr.controll(init_result)){
				struct_save();
				break;
			}				
			if(pre_params.show_freq>0&&ctr.total_rounds%pre_params.show_freq==pre_params.show_freq-1){
				show_compare();
				show_compare('t');
			}
			if(ctr.total_rounds%ctr.save_freq==ctr.save_freq-1){
				struct_save();
				save_params();
			}*/
			if(cnt%1500==1499)struct_save();
			ctr.real_rounds=5;
			ctr.rounds_init=1;
			//ctr.rounds_init++;
			all_bl=false;
			//all_group_num++;
			all_group_idx=0;
		}
		struct_save();
		save_params();
	}

	virtual bool show_and_control(int i){
		//if(i==0||ctr.show_mod=='1')coutd<<i<<" rlt"<<multi_perceptrons::result;
		if(i==0)init_result=multi_perceptrons::result;
		return true;
	}
	virtual bool pause_action(){
		return operate();
	}
	virtual void cacul(){
		double avg_result=0;
		double *avg_deriv;
		cudaMalloc((void **)&avg_deriv,sizeof(double)*weight_len);
		cudaMemset(avg_deriv,0,sizeof(double)*weight_len);
		int num=ctr.rounds_init;
		for(int i=0;i<num;i++){
			cacul_parallel_load();
			avg_result+=multi_perceptrons::result;
			array_float_plus_double(multi_perceptrons::deriv,avg_deriv,weight_len);
		}
		float alpha=1.0f/num;
		multi_perceptrons::result=avg_result*alpha;
		array_type_trans(avg_deriv,multi_perceptrons::deriv,weight_len);
		CUBT(cublasSscal(cublasHandle,weight_len, &alpha,multi_perceptrons::deriv,1));	
		cudaFree( avg_deriv);
		//show_compare();
		//show_compare('t');
	//	struct_save();

	}
	virtual void cacual_compare(char mod='c'){
		int n=multi_perceptrons::data_num;
		virtual_data_set::cmp_data &s=(mod=='c')?data_set->compare_data:data_set->train_cmp_data;
		set_data_num(s.num);
		s.result=get_result(s.trans_input,s.target);
		cudaMemcpy(s.output,nerv_top->output,sizeof(float)*multi_perceptrons::output_dimen*multi_perceptrons::data_num,cudaMemcpyDeviceToDevice);		
		CUDA_CHECK;
		set_data_num(n);
	}
};

void *pthread_train(void *t){
	multi_perceptrons_train *p=(multi_perceptrons_train *)t;
	//p->search_tool::search(p->ctr.real_rounds);
	//getchar();
	p->cacul_nerv(p->load_train_data_class::input,p->load_train_data_class::target);	
	return t;
	};

void *pthread_preload(void *t){	
	multi_perceptrons_train *p=(multi_perceptrons_train *)t;
	srand(time(NULL)+p->seeder);
	p->pre_load(p->ctr.data_num);
	return t;
	};


/*
class deep_train{
public:
	multi_perceptrons *main;
	multi_perceptrons *train;
	deep_nerv::train_set_struct train_set;
	string path;
	string train_path;
	struct record{
		char mod;
		int layer_num;
		int from;
		float dev;
		record(){
			mod='l';
			layer_num=1;
			from=1;		
		}
		void load(string path){			
			ifstream fin(path,ios::binary);
			if(!fin)set(path);
			else{
				cout<<endl<<"正在读取训练参数"<<path;
				fin.read((char *)&mod,sizeof(record));				
			}
			show();
		}
		void save(string path){
			cout<<endl<<"正在保存训练参数"<<path;
			ofstream fin(path,ios::binary);
			fin.write((char *)&mod,sizeof(record));		
		}
		void show(){
			cout<<endl<<"\t1.模式:[ l:逐层训练,n:监督调优,u:无监督调优 ]"<<mod;
			cout<<endl<<"\t2.训练层数:"<<layer_num;
			cout<<endl<<"\t3.起始隐层:"<<from;
			cout<<endl<<"\t4.终止拟合偏差:"<<dev;
		}
		void set(string path){
			show();
			cout<<endl<<"\tc.确定 l.查看 ";			
				string s;
				do{
					cout<<endl<<">>>";
					cin>>s;
					switch(s[0]){
					case '1':
						cin>>mod;
						break;
					case '2':
						cin>>layer_num;
						break;
					case '3':
						cin>>from;
						break;	
					case '4':
						cin>>dev;
						break;	
					case 'l':
						show();
						cout<<endl<<"\tc.确定 l.查看";			
						break;					
					}
				}while(s[0]!='c');
			save(path);
		}
	};
	record train_record;
	deep_train(){
		train=NULL;
	}
	void load_data(float *&input,int data_num,float *&target,char mod){		
		main->pre_load_train_data(mod);	
		deep_nerv::data &t=((mod=='t')?main->trn:main->cmp);
		if(train_record.from>1){
			main->set_data_num(data_num);
			main->run(t.input_data,input,train_record.from-1,data_num,true,true);
		}
		else {
			cudaMemcpy(input,t.input_data,sizeof(float)*train->input_dimen*data_num,cudaMemcpyDeviceToDevice);
			CUDA_CHECK;
		}
		switch (train_record.mod){
		case 'u':
			cudaMemcpy(target,t.input_data,sizeof(float)*train->output_dimen*data_num,cudaMemcpyDeviceToDevice);
			CUDA_CHECK;
			break;
		case 'n':
			cudaMemcpy(target,t.target_data,sizeof(float)*train->output_dimen*data_num,cudaMemcpyDeviceToDevice);
			CUDA_CHECK;
			break;
		case 'l':
			target=input;
			break;
		}		
	}
	void layer_train(){
		if(train!=NULL)delete train;
		nerv *nervs=new nerv[train_record.layer_num];
		for(int i=0;i<train_record.layer_num;i++)
			nervs[i]=*(main->nervs[train_record.from-1+i]);
		char l_path[200];
		sprintf(l_path,"%slayer_%c_%i_%i\\",path.c_str(),train_record.mod,train_record.from,train_record.layer_num);
		cout<<endl<<"当前训练数据路径:"<<l_path;
		train=new deep_nerv(l_path,nervs,main,train_record.layer_num);
		train->load_data->deep_train_class=(void *)this;
		train->train_set=train_set;
		train->pre_load_train_data('c');
		train->load_train_data();
		train->start();	
		train->save_struct();
		train_set.save((train_path+"nerv.trn"));
	
	}
	void run(string p){
		path=p;
		train_path=p+"layer\\";		
		main=new deep_nerv(path,'r');
		main->create_folder(train_path);
		train_record.load((train_path+"nerv.rcd"));
		train_set.load((train_path+"nerv.trn"));
		cout<<endl<<"<逐层训练>";
		cout<<endl<<"\t1.调整基本设置";
		cout<<endl<<"\t2.调整训练参数";
		cout<<endl<<"\tc.开始(继续)训练 e.退出";
		string s;
		do{
			cout<<endl<<">>";
			cin>>s;
			switch(s[0]){
			case '1':
				train_record.set((train_path+"nerv.rcd"));
				break;
			case '2':
				train_set.set((train_path+"nerv.trn"));
				break;
			case 'e':
				train_set.set((train_path+"nerv.trn"));
				break;
			}
		}while(s[0]!='c');
		if(train_set.compare_num==-1){
			main->pre_load_train_data('c');
			main->load_train_data();	
		for(int l=train_record.from;l<main->layers_num;l++){			
			layer_train();
			for(int i=0;i<train_record.layer_num;i++){
				cudaMemcpy(main->nervs[train_record.from-1+i]->weight_data
					,train->nervs[i]->weight_data
					,sizeof(float)*train->nervs[i]->weight_len
					,cudaMemcpyDeviceToDevice);
				CUDA_CHECK;
			}		
			main->save_struct();
			cout<<endl<<"已将训练结果移入主网络中";
			train_set.save((train_path+"nerv.trn"));
			train_record.from++;
			train_record.save((train_path+"nerv.rcd"));
		}

		}
	}
};
*/