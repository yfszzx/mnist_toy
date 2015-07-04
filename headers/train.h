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
			show_freq=50;
			confirm_interval=2000;
			z=3;
			last_count=0;
			rec=NULL;
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
			coutd<<"\t\t<ѵ������>";
			coutd<<"\t1.��ʼminibatch��С:"<<init_data_num;
			coutd<<"\t2.��ʼ��������:"<<init_iteration_num;/*[EDIT_SYMBOL]*/
			coutd<<"\t3.minibatch����ϵ��:"<<data_num_increase_rate;
			coutd<<"\t4.������������ϵ��:"<<iteration_num_decay_rate;
			coutd<<"\t5.����minibatch��С:"<<final_data_num;
			coutd<<"\t6.���յ�������:"<<final_iteration_num;
			coutd<<"\t7.��ǰminibatch��С:"<<data_num;	
			coutd<<"\t8.��ǰ��������:"<<iteration_num;			
			coutd<<"\t0.��ʾ���"<<show_freq;
			coutd<<"\ta.ȷ�ϼ��:"<<confirm_interval;
			coutd<<"\tb.ȷ��Zֵ����ʧ�������ٵ������̶�):"<<z;
			coutd<<"\td.��ʾ��ʽ:0.����ʾ���� 1.��ʾ�������"<<show_mod;
		
			
			coutd<<"\ts.������:"<<save_freq;
		}
		void set(){
			char sel;
			show();
			coutd<<"\tc.ȷ�� l.�鿴";
			while(1){
				coutd<<"���ò���>";
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
				case '6':
					cin>>final_iteration_num;					
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
		}
		void controll(float loss,float avg_step=-1){
			//��¼�������*������
			cacul_sample_num+=(int)data_num*(cacul_count-last_count);
			last_count=cacul_count;

			//��¼�����̶ȣ������������̶ȵ�������
			int ci=confirm_interval;
			rec[confirm_round%ci]=loss;				
			if(confirm_round%ci==ci-1){
				float avg=0;
				for(int i=0;i<(int)ci;i++){
					avg+=rec[i];
				}
				avg/=(int)ci;
				float dev=0;
				for(int i=0;i<(int)ci;i++){
					dev+=(rec[i]-avg)*(rec[i]-avg);
				}
				dev/=(int)ci;
				if(confirm_round>ci){//z����
					float zz=(avg-confirm_last_avg)/sqrt((dev+confirm_last_dev)/ci);
					if(zz>-z){
						data_num*=data_num_increase_rate;
						//z/=sqrt(data_num_increase_rate);
						iteration_num*=iteration_num_decay_rate;
						if(data_num>final_data_num)data_num=final_data_num;
						if(iteration_num<final_iteration_num)iteration_num=final_iteration_num;											
					}
				//	coutd<<"���"<<confirm_round<<"��ƽ��loss:"<<avg<<"ǰ"<<confirm_round<<"��ƽ��loss:"<<confirm_last_avg<<" Z�������̶�)"<<(-zz);//[EDIT_SYMBOL]
				//	coutd<<"\n";//[EDIT_SYMBOL]
					coutd<<"\nloss���������̶�"<<(-zz);
					confirm_round=-1;
				}		
				
				confirm_last_avg=avg;
				confirm_last_dev=dev;
			}
			confirm_round++;


			if(total_rounds%show_freq==show_freq-1){
				float t=(float)(clock()-round_start)/CLOCKS_PER_SEC;			
				total_time+=t;
				int n=cacul_count-cacul_count_last;
				cacul_count_last=cacul_count;
				coutd<<"<��"<<total_rounds
					<<"��("<<cacul_count<<"),"
					<<n<<"�μ��㣬��ʱ"
					<<t<<"��(����"
					<<setprecision(3)<<(t/n)<<"��),"<<setprecision(6)
					<<"����"<<(int)total_time/60<<"�� ";
				
				coutd<<"����"<<iteration_num<<" minibatch"<<data_num;
				if(avg_step>0)cout<<" ���� "<<setprecision(4)<<avg_step<<setprecision(6);//[EDIT_SYMBOL]
				round_start=clock();
			}		
			
			total_rounds++;			
			
		}

	};
void *pthread_train(void *);
void *pthread_preload(void *);
class multi_perceptrons_train: public multi_perceptrons,public search_tool{
private:
			
	bool read_params(){
		string path=root+"params.stl";
		ifstream fin(path,ios::binary);
		if(!fin)return false;
		coutd<<"���ڶ�ȡ"<<path;
		fin.read((char *)&ctr,sizeof(optimization_controller)-sizeof(float *));
		fin.close();
		return true;
	}

	virtual bool show_and_control(int i){
		if(ctr.show_mod=='1')coutd<<i<<" loss"<<multi_perceptrons::result<<" gradient"<<length(multi_perceptrons::deriv)<<" step"<<current_step;
		if(i==0)init_result=multi_perceptrons::result;
		return true;
	}
	virtual void cacul(){
		cacul_nerv(data_input,data_target);	
		ctr.cacul_count++;
		
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
		ctr.controll(init_result,current_step);
		if(ctr.total_rounds%ctr.save_freq==ctr.save_freq-1){
			struct_save(ctr.cacul_count);
			save_params();
		}
		if(pause_flag&&!operate())return false;
		return true;
	}
	void train_bigenning(){
		set_data_num(ctr.data_num);
		pre_load(ctr.data_num);		
		load_data();
	}
public:
	int seeder;
	float init_result;
	optimization_controller ctr;
	float * data_input;
	float * data_target;
	string root;
	multi_perceptrons_train(string name,char mod='t'):multi_perceptrons(name),search_tool(name) {
		//mod��t train r read b bagging c check
		g_gpu_init();
		root=name;
		if(mod=='t'||mod=='b')train_mod=true;
	}
	void save_params(){
		string path=root+"params.stl";
		coutd<<"���ڱ���"<<path;
		ofstream fin(path,ios::binary);
		fin.write((char *)&ctr,(sizeof(optimization_controller)-sizeof(float *)));
		fin.close();
		save_search();
	}
	
	void mlp_init(int i_d,int o_d){
		input_dimen=i_d;
		output_dimen=o_d;		
		if(!struct_read())struct_set(input_dimen,output_dimen);	
		if(input_dimen!=nervs[0]->input_dimen||output_dimen!=nervs[layers_num-1]->nodes_num){
			coutd<<"����������ṹ�������ļ���һ�£���������������ṹ";
			struct_set(input_dimen,output_dimen);
		}
	}
	void train_init(){
		search_init(weight_len,&(multi_perceptrons::result),weight,multi_perceptrons::deriv);
		if(!read_params()){
			ctr.set();
			save_params();
		}		
		ctr.init();	
		train_bigenning();
	}

	void train_reset(){
		multi_perceptrons::reset();
		ctr.reset();
		reset_step();
		train_bigenning();
	}
	void train(int end_num=0){
		while(1){	
			cacul_parallel_load();
			show_and_record();
			if(!train_controll())return;
			if(end_num>0&&ctr.total_rounds>end_num)break;
		}
		struct_save(ctr.cacul_count);
		save_params();
	}
	virtual void show_and_record()=0;
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
public:
	void menu(){
		coutd;
		coutd<<"\t1.��������";
		coutd<<"\t2.���³�ʼ��";
		coutd<<"\t3.����ѵ������";
		coutd<<"\t4.������������";
		coutd<<"\t5.����Ԥ�������";
		coutd<<"\t6.������";		
		coutd<<"\t7."<<"����Ԥ��������";	
		coutd<<"\t8."<<data_set->action_illu;
		coutd<<"\tc.��ʼ(����)ѵ�� l.�˵� e.�˳�";
		coutd<<"\t(���������м���'p'����ʾ���˵�)";
	}
	virtual bool operate(){		
		while(1){
			menu();
			char s;
			coutd<<"ѡ��>";
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
				 multi_perceptrons_train::save_params();
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
		if(in_cuda_pos){
			pre_input=s_input;
		}else{
			cudaMalloc((void**)&pre_input,sizeof(float)*pre_dimen*num);
			CUDA_CHECK;
			cudaMemcpy(pre_input,s_input,sizeof(float)*pre_dimen*num,cudaMemcpyHostToDevice);
			CUDA_CHECK;

		}
		cudaMalloc((void**)&input,sizeof(float)*multi_perceptrons::input_dimen*num);
		trans_data(pre_input,input,num);
		float ret=multi_perceptrons::run(input,out,s_target,num,dropout,true,out_cuda_pos);
		if(!in_cuda_pos)cudaFree(input);
		return ret;
	}
	float *layer_out(int layer,float *s_input,int num,bool in_cuda_pos=true){//�м�������������м��������ݵ�ָ��(device)
		float *pre_input;
		float *input;
		if(in_cuda_pos){
			pre_input=s_input;
		}else{
			cudaMalloc((void**)&pre_input,sizeof(float)*pre_dimen*num);
			CUDA_CHECK;
			cudaMemcpy(pre_input,s_input,sizeof(float)*pre_dimen*num,cudaMemcpyHostToDevice);
			CUDA_CHECK;

		}
		cudaMalloc((void**)&input,sizeof(float)*multi_perceptrons::input_dimen*num);
		trans_data(pre_input,input,num);
		float *ret=multi_perceptrons::layer_out(layer,s_input,num,in_cuda_pos);
		if(!in_cuda_pos)cudaFree(input);
		return ret;
	}
	main_train(string name,DATA_CLASS *dt,char mod='t'):multi_perceptrons_train(name,mod),load_train_data_class(dt,name){
		//mod��t train r read b bagging c check
		mlp_init(load_train_data_class::input_dimen,load_train_data_class::output_dimen);
		if(mod=='r')return;//read_only
	
		train_init();	
		show_and_record_init(ctr.total_rounds);
		if(mod=='t'){
			if(!operate())return;//train;
			train();		
		}
		
	}
	virtual void load_data(){
		load_train_data_class::load_data();	
		data_input=load_train_data_class::input;
		data_target=load_train_data_class::target;
	};
	virtual void pre_load(int num){
		load_train_data_class::pre_load(num);
	};		
	virtual void show_and_record(){
		avg_loss+=init_result;
		avg_step+=current_step;
		if(pre_params.show_freq>0&&ctr.total_rounds%pre_params.show_freq==pre_params.show_freq-1){
			load_train_data_class::show_and_record(ctr.total_time,ctr.cacul_count,ctr.cacul_sample_num);
			set_data_num(ctr.data_num);
		}
	
	};
	virtual void cacual_compare(char mod='c'){
		virtual_data_set::cmp_data &s=(mod=='c')?data_set->compare_data:data_set->train_cmp_data;
		set_data_num(s.num);
		s.result=get_result(s.trans_input,s.target);
		cudaMemcpy(s.output,nerv_top->output,sizeof(float)*multi_perceptrons::output_dimen*multi_perceptrons::data_num,cudaMemcpyDeviceToDevice);		
		CUDA_CHECK;
	}
		
};


