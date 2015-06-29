
class load_train_data_class:public PCA{
public:
	struct data_pre_params{
		float noise_scale;
		int tmp;
		int show_freq;
		int compare_num;
		void show(){
			coutd<<"<数据预处理参数>";
			coutd<<"\t1.高斯噪音尺度"<<noise_scale;
			coutd<<"\t2.对照数据数量"<<compare_num;
			coutd<<"\t3.对照数据显示间隔"<<show_freq;
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
				case '3':
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
			show_freq=500;
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
	load_train_data_class(DATA_CLASS *p,string path):PCA(path){		
		input=NULL;
		target=NULL;
		get_input=NULL;
		get_target=NULL;
		//all_input=NULL;
		//all_target=NULL;
		//all_bl=false;
		//all_group_idx=0;
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
	float show_compare(char mod='c'){
		if(pre_params.compare_num==0)return 0;		
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
		return s.result;
		
	}
};
void *pthread_train(void *);
void *pthread_preload(void *);
class multi_perceptrons_train: public multi_perceptrons,public search_tool,public load_train_data_class{
public:
	string root;
	int seeder;
	int input_dimen;
	int output_dimen;
	float init_result;
	optimization_controller ctr;
	multi_perceptrons_train(string name,DATA_CLASS *dt,char mod='t'):multi_perceptrons(name),load_train_data_class(dt,name){
		//mod：t train r read b bagging c check
		g_gpu_init();
		root=name;
		input_dimen=load_train_data_class::input_dimen;
		output_dimen=load_train_data_class::output_dimen;
		if(mod=='t'||mod=='b')train_mod=true;
		if(!struct_read())struct_set(input_dimen,output_dimen);
	
		if(mod=='r')return;//read_only
		if(input_dimen!=nervs[0]->input_dimen||output_dimen!=nervs[layers_num-1]->nodes_num){
			coutd<<"【错误】网络结构与数据文件不一致，请重新设置网络结构";
			struct_set(input_dimen,output_dimen);
		}
		search_init(weight_len,&(multi_perceptrons::result),weight,multi_perceptrons::deriv);
		if(!read_params()){
			ctr.set();
			set_search();
			//pre_params.set();//[EDIT_SYMBOL]
			save_params();
		}		
		ctr.init();	
		search_tool::set_init();
		if(mod=='d')return;//dropout;
		if(mod=='b'){//bagging;
			train_reset();
		}
		if(mod=='t')if(!operate())return;//train;
		train();
	
		
	}
	bool read_params(){
		string path=root+"params.stl";
		ifstream fin(path,ios::binary);
		if(!fin)return false;
		coutd<<"正在读取"<<path;
		fin.read((char *)&(search_tool::set),sizeof(search_set));	
		search_tool::set_init();
		fin.read((char *)&ctr,sizeof(optimization_controller)-sizeof(float *));
		fin.read((char *)&pre_params,sizeof(data_pre_params));
		fin.close();
		return true;
	}
	void save_params(){
		string path=root+"params.stl";
		coutd<<"正在保存"<<path;
		ofstream fin(path,ios::binary);
		fin.write((char *)&(search_tool::set),sizeof(search_set));
		fin.write((char *)&ctr,(sizeof(optimization_controller)-sizeof(float *)));
		fin.write((char *)&pre_params,sizeof(data_pre_params));
		fin.close();
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
	void menu(){
		coutd;
		coutd<<"\t1.设置网络";
		coutd<<"\t2.重新初始化";
		coutd<<"\t3.设置训练参数";
		coutd<<"\t4.设置搜索参数";
		//coutd<<"\t5.设置预处理参数";[EDIT_SYMBOL]
		coutd<<"\t6.保存结果";		
		//coutd<<"\t7."<<"重新预处理数据";	[EDIT_SYMBOL]	
		coutd<<"\t8."<<data_set->action_illu;
		coutd<<"\tc.开始(继续)训练 l.菜单 e.退出";
		coutd<<"\t(程序运行中键入'p'可显示本菜单)";
	}
	bool operate(){
		
		while(1){
			menu();
			char s;
			coutd<<"选择>";
			cin>>s;
			switch(s){
			case '1':
				struct_edit();
				search_init(weight_len,&(multi_perceptrons::result),weight,multi_perceptrons::deriv);
				search_tool::set_init();
				break;
			case '2':
				train_reset();
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
				 reset_pca();
				break;
			case '8':
				data_set->self_action(this);
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
	void train_reset(){
		reset();
		ctr.reset();
		train_bigenning();
	}
	void train_bigenning(){
//		all_group_num=ctr.rounds_init;
		show_and_record_init();
		set_data_num(ctr.data_num);
		pre_load(ctr.data_num);		
		load_data();
		bl_exit=false;
		
		
	//	average_mlp.init(weight_len,100,weight);
		//	dropout_trans_weight();
	}
	void train_controll(){
		ctr.controll(init_result,search_tool::set.step);
		avg_loss+=init_result;
		avg_step+=search_tool::set.step;
		if(pre_params.show_freq>0&&ctr.total_rounds%pre_params.show_freq==pre_params.show_freq-1){
			show_and_record();
		}
		if(ctr.total_rounds%ctr.save_freq==ctr.save_freq-1){
			struct_save(ctr.cacul_count);
			save_params();
		}
	}
	void show_and_record_init(){
		record_path=root+"train_record.csv";
		if(ctr.total_rounds==0){
			record_file=new ofstream(record_path);
			(*record_file)<<"时间,迭代轮次,计算次数,loss,对照值,采样值,步长"<<endl;
		}else{
			record_file=new ofstream(record_path,ios::app);
			(*record_file)<<ctr.cacul_count<<"重新载入"<<endl;
		}
		avg_loss=0;
		avg_step=0;
		//show_and_record();
	}
	void show_and_record(){
		(*record_file)<<ctr.total_time<<","<<ctr.cacul_count<<","<<ctr.cacul_sample_num<<","<<(avg_loss/pre_params.show_freq)<<",";
		(*record_file)<<show_compare()<<",";
		(*record_file)<<show_compare('t')<<",";
		(*record_file)<<(avg_step/pre_params.show_freq)<<endl;
		avg_loss=0;
		avg_step=0;
		set_data_num(ctr.data_num);
	}

	struct {
		double *average_weight;
		float *init_weight;
		int count;
		int record_num;
		int weight_len;
		void init(int w_len,int r_num,float *weight){
			weight_len=w_len;
			record_num=r_num;
			count=0;
			cudaMalloc((void ** )&average_weight,sizeof(double)*weight_len);
			cudaMalloc((void ** )&init_weight,sizeof(float)*weight_len);
			cudaMemset(average_weight,0,sizeof(double)*weight_len);
			cudaMemcpy(init_weight,weight,sizeof(float)*weight_len,cudaMemcpyDeviceToDevice);
			
		}
		void record(float *weight){
			array_float_plus_double(weight,average_weight,weight_len);			
			if(count==record_num-1){
				//array_type_trans(average_weight,init_weight,weight_len);
				array_type_trans(average_weight,weight,weight_len);
				float alpha=1.0f/record_num;
				//CUBT(cublasSscal(cublasHandle,weight_len, &alpha,init_weight, 1));
				CUBT(cublasSscal(cublasHandle,weight_len, &alpha,weight, 1));
				cudaMemset(average_weight,0,sizeof(double)*weight_len);
				coutd<<"reset wight";
				
			}
			//cudaMemcpy(weight,init_weight,sizeof(float)*weight_len,cudaMemcpyDeviceToDevice);
			count++;
			count%=record_num;
		}
	}average_mlp;



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
		//	average_mlp.record(weight);
	}


	void train(){
		train_bigenning();
		while(1){	
			//search_tool::search(ctr.iteration_num);
			cacul_parallel_load();
			train_controll();
			if(bl_exit)return;
		}
		struct_save(ctr.cacul_count);
		save_params();
	}

	virtual bool show_and_control(int i){
		if(ctr.show_mod=='1')coutd<<i<<" loss"<<multi_perceptrons::result<<" gradient"<<length(multi_perceptrons::deriv)<<" step"<<search_tool::set.step;
		if(i==0)init_result=multi_perceptrons::result;
		return true;
	}
	virtual bool pause_action(){
		bl_exit=!operate();
		return bl_exit;
	}
	virtual void cacul(){
		cacul_nerv(load_train_data_class::input,load_train_data_class::target);	
		//cacul_parallel_load();
		ctr.cacul_count++;
		/*double avg_result=0;
		double *avg_deriv;
		cudaMalloc((void **)&avg_deriv,sizeof(double)*weight_len);
		cudaMemset(avg_deriv,0,sizeof(double)*weight_len);
		int num=10;//ctr.rounds_init;
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
	//	struct_save();*/

	}
	virtual void cacual_compare(char mod='c'){
		virtual_data_set::cmp_data &s=(mod=='c')?data_set->compare_data:data_set->train_cmp_data;
		set_data_num(s.num);
		s.result=get_result(s.trans_input,s.target);
		cudaMemcpy(s.output,nerv_top->output,sizeof(float)*multi_perceptrons::output_dimen*multi_perceptrons::data_num,cudaMemcpyDeviceToDevice);		
		CUDA_CHECK;
	}
	private:
		string record_path;
		ofstream *record_file;
		float avg_loss;
		float avg_step;
		bool bl_exit;
		
};

void *pthread_train(void *t){
	multi_perceptrons_train *p=(multi_perceptrons_train *)t;
	p->search_tool::search(p->ctr.iteration_num);
	//p->cacul_nerv(p->load_train_data_class::input,p->load_train_data_class::target);	
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