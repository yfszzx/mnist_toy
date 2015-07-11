struct optimization_controller{
		int init_iteration_num;
		int init_data_num;
		float data_num_increase_rate;
		float iteration_num_decay_rate;
		int final_data_num;
		int final_iteration_num;
		float data_num;
		float iteration_num;
		int cacul_count;
		int cacul_count_last;
		int total_rounds;
		int cacul_sample_num;
		int last_count;
		char show_mod;	
		int save_freq;
		clock_t start;
		clock_t round_start;
		float total_time;		
		float confirm_last_avg;
		float confirm_last_dev;
		int confirm_round;
		int confirm_interval;
		float z;
		int show_freq;
		int snum;
		char file_name[50];
		float *rec;		
		optimization_controller(){
			init_iteration_num=1;
			init_data_num=1000;
			data_num_increase_rate=1;
			iteration_num_decay_rate=1;
			final_data_num=1000;
			final_iteration_num=1;
			data_num=init_data_num;
			iteration_num=init_iteration_num;
			save_freq=2000;
			cacul_count=0;
			round_start=0;
			total_time=0;
			total_rounds=0;
			cacul_sample_num=0;
			show_mod='0';	
			show_freq=500;
			confirm_interval=2000;
			z=3;
			last_count=0;
			rec=NULL;
			sprintf(file_name,"%s","train_controll.stl");
		}
		~optimization_controller(){
			safe_free(rec);
		}
		void reset(){
			cacul_count=0;
			round_start=0;
			total_time=0;
			total_rounds=0;
			cacul_sample_num=0;
			iteration_num=init_iteration_num;			
			data_num=init_data_num;
			init();
		}
		void show(){
			coutd<<"\t\t<训练参数>";
			coutd<<"\t1.初始minibatch大小:"<<init_data_num;
			coutd<<"\t2.初始迭代次数:"<<init_iteration_num;/*[EDIT_SYMBOL]*/
			coutd<<"\t3.minibatch增涨系数:"<<data_num_increase_rate;
			coutd<<"\t4.迭代次数减退系数:"<<iteration_num_decay_rate;
			coutd<<"\t5.最终minibatch大小:"<<final_data_num;
			coutd<<"\t7.当前minibatch大小:"<<data_num;	
			coutd<<"\t8.当前迭代次数:"<<iteration_num;			
			coutd<<"\t0.记录间隔"<<show_freq;
			coutd<<"\ta.确认间隔:"<<confirm_interval;
			coutd<<"\tb.确认Z值（损失函数减少的显著程度):"<<z;
			coutd<<"\td.显示方式:0.不显示过程 1.显示计算过程"<<show_mod;
		
			
			coutd<<"\ts.保存间隔:"<<save_freq;
		}
		void set(){
			char sel;
			show();
			coutd<<"\tc.确定 l.查看";
			while(1){
				coutd<<"设置参数>";
				cin>>sel;
				switch(sel){
				case '1':
					cin>>init_data_num;
					data_num=init_data_num;
					break;
				case '2':
					cin>>init_iteration_num;
					iteration_num=init_iteration_num;
					break;
				case '3':
					cin>>data_num_increase_rate;
					break;
				case '4':
					cin>>iteration_num_decay_rate;
					break;
				case '5':
					cin>>final_data_num;
					break;
				case '7':
					cin>>data_num;
					break;
				case '8':					
					cin>>iteration_num;
					break;
				case 'd':
					cin>>show_mod;
					break;
				case '0':
					cin>>show_freq;
					snum=show_freq/10;
					break;
				case 'a':
					cin>>confirm_interval;
					break;
				case 'b':
					cin>>z;
					break;
				case 's':
					cin>>save_freq;
					break;
				case 'l':
					show();
					break;
				case 'c':
					return;
			};
		}
	}
	
		void init(){
			round_start=clock();
			start=clock();
			cacul_count_last=cacul_count;
			last_count=cacul_count;
			confirm_round=0;
			safe_free(rec);
			rec=new float[confirm_interval];
			snum=show_freq/10;
		}
		char controll(float loss){//返回'n'：正常 's'显示参数 'e'结束训练
			//记录计算次数*样本数
			cacul_sample_num+=(int)data_num*(cacul_count-last_count);
			last_count=cacul_count;

			//记录显著程度，并根据显著程度调整参数		
			rec[confirm_round%confirm_interval]=loss;				
			if(confirm_round%confirm_interval==confirm_interval-1){
				int ci=confirm_interval;
				float avg=0;
				for(int i=0;i<ci;i++){
					avg+=rec[i];
				}
				avg/=ci;
				float dev=0;
				for(int i=0;i<ci;i++){
					dev+=(rec[i]-avg)*(rec[i]-avg);
				}
				dev/=(int)ci;
				if(confirm_round>ci){//z检验
					float zz=(avg-confirm_last_avg)/sqrt((dev+confirm_last_dev)/ci);
					if(zz>0&&data_num==final_data_num&&iteration_num==final_iteration_num)return 'e';
					if(zz>-z){
						data_num*=data_num_increase_rate;
						//z/=sqrt(data_num_increase_rate);						
						if(data_num>final_data_num)data_num=final_data_num;
						iteration_num*=iteration_num_decay_rate;
						if(iteration_num<final_iteration_num)iteration_num=final_iteration_num;											
					}
				//	coutd<<"最近"<<confirm_round<<"次平均loss:"<<avg<<"前"<<confirm_round<<"次平均loss:"<<confirm_last_avg<<" Z（显著程度)"<<(-zz);//[EDIT_SYMBOL]
				//	coutd<<"\n";//[EDIT_SYMBOL]
					coutd<<"\nloss减少显著程度"<<(-zz);
					confirm_round=-1;
				}		
				
				confirm_last_avg=avg;
				confirm_last_dev=dev;
			}
			confirm_round++;
			total_rounds++;	
			if(total_rounds%snum==snum-1){
				float t=(float)(clock()-round_start)/CLOCKS_PER_SEC;			
				total_time+=t;
				int n=cacul_count-cacul_count_last;
				cacul_count_last=cacul_count;
				coutd<<"第"<<total_rounds
					<<"轮("<<cacul_count<<"),"
					<<n<<"次计算，用时"
					<<t<<"秒(单次"
					<<setprecision(3)<<(t/n)<<"秒),"<<setprecision(6)
					<<"共计"<<(int)total_time/60<<"分 ";						
				round_start=clock();
				return 's';
			}		
			return 'n';
					
			
		}
		void save(string path){
			path+=file_name;
			coutd<<"正在保存"<<path;
			ofstream fin(path,ios::binary);
			fin.write((char *)this,(sizeof(optimization_controller)-sizeof(float *)));
			fin.close();
		}
		bool read(string path){
			path+=file_name;
			ifstream fin(path,ios::binary);
			if(!fin)return false;
			coutd<<"正在读取"<<path;
			fin.read((char *)this,sizeof(optimization_controller)-sizeof(float *));
			fin.close();
			return true;
		}

	};
void *pthread_train(void *);
void *pthread_preload(void *);
class multi_perceptrons_train: public multi_perceptrons,public search_tool{
private:
	float init_result;	
	virtual bool show_and_control(int i){
		if(ctr.show_mod=='1')coutd<<i<<" loss"<<multi_perceptrons::result<<" gradient"<<init_deriv<<" step"<<current_step;
		if(i==0)init_result=multi_perceptrons::result;
		return true;
	}
	virtual void cacul(){
		cacul_nerv(data_input,data_target);	
		ctr.cacul_count++;
		/*for(int i=layers_num-1;i>=0;i--){
			coutd<<array_length(nervs[i]->deriv,nervs[i]->weight_len);
		}
		coutd<<"deriv"<<array_length(multi_perceptrons::deriv,weight_len);
		*/
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
			load_data();
	}
	bool train_controll(){
		char s=ctr.controll(init_result);
		if(s=='s'){
				coutd<<"次数"<<ctr.iteration_num<<" minibatch"<<data_num;	
				cout<<" loss "<<setprecision(4)<<init_result<<" 步长 "<<current_step<<setprecision(6)<<" 梯度 "<<init_deriv;
		}
		if(s=='e')return false;
		if(ctr.total_rounds%ctr.save_freq==ctr.save_freq-1){
			struct_save(ctr.cacul_count);
			save_params();
		}		
		show_and_record();
		if(pause_flag&&!operate())return false;
		return true;
	}

	void train_bigenning(){
		show_and_record_init();
		set_data_num(ctr.data_num);
		pre_load(ctr.data_num);		
		load_data();

	}
	void show_and_record_init(){
		record_path=root+"train_record.csv";
		if(ctr.total_rounds==0){
			record_file=new ofstream(record_path);
			(*record_file)<<"时间,迭代轮次,计算次数,loss,对照值,采样值,步长"<<endl;
		}else{
			record_file=new ofstream(record_path,ios::app);
			(*record_file)<<ctr.total_rounds<<",重新载入"<<endl;
		}
		avg_loss=0;
		avg_step=0;		
	}
	void show_and_record(){
		avg_loss+=init_result;
		avg_step+=current_step;
		if(ctr.show_freq>0&&ctr.total_rounds%ctr.show_freq==ctr.show_freq-1){
		(*record_file)<<ctr.total_time<<","<<ctr.cacul_count<<","<<ctr.cacul_sample_num<<","<<(avg_loss/ctr.show_freq)<<",";
		(*record_file)<<show_compare()<<",";
		(*record_file)<<show_compare('t')<<",";
		(*record_file)<<(avg_step/ctr.show_freq)<<endl;
		avg_loss=0;
		avg_step=0;
		set_data_num(ctr.data_num);
		}
	}
	string record_path;
	ofstream *record_file;
	float avg_loss;
	float avg_step;
public:
	int seeder;
	float * data_input;
	float * data_target;
	string root;
	optimization_controller ctr;
	multi_perceptrons_train(string name,char mod='t'):multi_perceptrons(name),search_tool(name) {
		//mod：t train r read b bagging c check
		g_gpu_init();
		root=name;
		if(mod=='t'||mod=='b')train_mod=true;
	}
	void save_params(){
		ctr.save(root);
		save_search();
	}
	
	void mlp_init(int i_d,int o_d){
		input_dimen=i_d;
		output_dimen=o_d;		
		if(!struct_read())struct_set(input_dimen,output_dimen);	
		if(input_dimen!=nervs[0]->input_dimen||output_dimen!=nervs[layers_num-1]->nodes_num){
			coutd<<"【错误】网络结构与数据文件不一致，请重新设置网络结构";
			struct_set(input_dimen,output_dimen);
		}
	}
	void mlp_simple_init(int in_dimen,int out_dimen,int nodes,char nodes_mod,char out_mod,char loss='2',float  decay=0,char decay_mod='2'){
		 if(!struct_read())struct_simple_set(in_dimen,out_dimen,nodes,nodes_mod,out_mod,loss,decay,decay_mod);
		 struct_save();
	}
	void train_init(){
		search_init(weight_len,&(multi_perceptrons::result),weight,multi_perceptrons::deriv);
		if(!ctr.read(root)){
			ctr.set();
			save_params();
		}		
		ctr.init();	
		train_bigenning();
	}

	void train_reset(){
		multi_perceptrons::reset();
		search_init(weight_len,&(multi_perceptrons::result),weight,multi_perceptrons::deriv);
		ctr.reset();
		reset_step();
		train_bigenning();
	}
	void train(int end_num=0){
		while(1){	
			cacul_parallel_load();
			if(!train_controll())return;
			if(end_num>0&&ctr.total_rounds>end_num)break;
		}
		struct_save(ctr.cacul_count);
		save_params();
	}
	
	virtual float show_compare(char mod='c')=0;
	virtual void pre_load(int)=0;
	virtual void load_data()=0;
	virtual bool operate()=0;


};
void *pthread_train(void *t){
	multi_perceptrons_train *p=(multi_perceptrons_train *)t;
	p->search(p->ctr.iteration_num);
	return t;
};

void *pthread_preload(void *t){	
	multi_perceptrons_train *p=(multi_perceptrons_train *)t;
	srand(time(NULL)+p->seeder);
	p->pre_load(p->ctr.data_num);
	return t;
};


class main_train:public  load_train_data_class,public  multi_perceptrons_train{
private:
//float *layerout_input;
//int layerout_num;
public:
	void menu(){
		coutd;
		coutd<<"\t1.设置网络";
		coutd<<"\t2.重新初始化";
		coutd<<"\t3.设置训练参数";
		coutd<<"\t4.设置搜索参数";
		coutd<<"\t5.设置预处理参数";
		coutd<<"\t6.保存结果";		
		coutd<<"\t7."<<"重新预处理数据";	
		coutd<<"\t8."<<data_set->action_illu;
		coutd<<"\tc.开始(继续)训练 l.菜单 e.退出";
		coutd<<"\t(程序运行中键入'p'可显示本菜单)";
	}

	void reset_pca(){
		load_train_data_class:: reset_pca();
		if(multi_perceptrons_train::input_dimen!=PCA::set.pca_dmn){
			struct_set(load_train_data_class::input_dimen,load_train_data_class::output_dimen);
			search_init(weight_len,&(multi_perceptrons::result),weight,multi_perceptrons::deriv);
			train_reset();
		}
	}
	float run(float *s_input,float *out,float *s_target,int num,float dropout=1.0f,bool in_cuda_pos=true,bool out_cuda_pos=true){
		float *pre_input;
		float *input;
		float *target;
		if(in_cuda_pos){
			pre_input=s_input;
		}else{
			cudaMalloc((void**)&pre_input,sizeof(float)*pre_dimen*num);
			CUDA_CHECK;
			cudaMemcpy(pre_input,s_input,sizeof(float)*pre_dimen*num,cudaMemcpyHostToDevice);
			CUDA_CHECK;

		}
		if(out_cuda_pos){
			target=s_target;
		}else{
			cudaMalloc((void**)&target,sizeof(float)*multi_perceptrons::output_dimen*num);
			CUDA_CHECK;
			cudaMemcpy(target,s_target,sizeof(float)*multi_perceptrons::output_dimen*num,cudaMemcpyHostToDevice);
			CUDA_CHECK;

		}
		cudaMalloc((void**)&input,sizeof(float)*multi_perceptrons::input_dimen*num);
		trans_data(pre_input,input,num);
		float ret=multi_perceptrons::run(input,out,target,num,dropout,true,out_cuda_pos);
		if(!in_cuda_pos){
			cudaFree(pre_input);
		}
		if(!out_cuda_pos){
			cudaFree(target);
		}
		cudaFree(input);
		CUDA_CHECK;
		return ret;
	}
/*	float *layer_out(int layer,float *s_input,int num,bool in_cuda_pos=true){//中间层的输出，返回中间层输出数据的指针(device)
		float *pre_input;
		if(in_cuda_pos){
			pre_input=s_input;
		}else{
			cudaMalloc((void**)&pre_input,sizeof(float)*pre_dimen*num);
			CUDA_CHECK;
			cudaMemcpy(pre_input,s_input,sizeof(float)*pre_dimen*num,cudaMemcpyHostToDevice);
			CUDA_CHECK;

		}
		if(layerout_num!=num){
			safe_gpu_free(layerout_input);
			cudaMalloc((void**)&layerout_input,sizeof(float)*multi_perceptrons::input_dimen*num);
			CUDA_CHECK;
			layerout_num=num;
		}
		trans_data(pre_input,layerout_input,num);
		CUDA_CHECK;
		float *ret=multi_perceptrons::layer_out(layer,layerout_input,num,in_cuda_pos);
		if(!in_cuda_pos)cudaFree(pre_input);
		return ret;
	}*/
	main_train(string name,void *dt,char mod='t'):multi_perceptrons_train(name,mod),load_train_data_class(dt,name){
		//mod：t train r read b bagging c check
		mlp_init(load_train_data_class::input_dimen,load_train_data_class::output_dimen);
		if(mod=='r')return;//read_only	
		//layerout_input=NULL;
		//layerout_num=0;
		train_init();
		get_compare_data();
		if(mod=='t'){
			if(!operate())return;//train;
			train();		
		}
		
	}
	/*~main_train(){
		safe_gpu_free(layerout_input);
	}*/
	virtual void load_data(){
		load_train_data_class::load_data();	
		data_input=load_train_data_class::input;
		data_target=load_train_data_class::target;
		CUDA_CHECK;
	};
	virtual void pre_load(int num){
		load_train_data_class::pre_load(num);
	};		
	virtual void cacual_compare(char mod='c'){
		cmp_data &s=(mod=='c')?data_set->compare_data:data_set->train_cmp_data;
		set_data_num(s.num);
		s.result=get_result(s.trans_input,s.target);
		cudaMemcpy(s.output,nerv_top->output,sizeof(float)*multi_perceptrons::output_dimen*multi_perceptrons::data_num,cudaMemcpyDeviceToDevice);		
		CUDA_CHECK;
	}
	virtual float show_compare(char mod='c'){
		return load_train_data_class::show_compare(mod);
	}
	virtual bool operate(){		
		while(1){
			menu();
			char s;
			coutd<<"选择>";
			cin>>s;
			switch(s){
			case '1':
				struct_edit();
				search_init(weight_len,&(multi_perceptrons::result),weight,multi_perceptrons::deriv);
				train_reset();
				break;
			case '2':
				train_reset();
				break;
			case '3':
				ctr.set();
				ctr.save(multi_perceptrons_train::root);
				break;
			case '4':
				set_search();
				break;
			case '5':
				pre_params.set();
				load_train_data_class::save_params();
				break;
			case '6':
				struct_save();
				 multi_perceptrons_train::save_params();
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
		
};


