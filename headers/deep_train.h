class MLP_layer_train:public multi_perceptrons_train{
public:
	main_train *main;
	int layer;
	cmp_data train_cmp_data;
	cmp_data compare_data;
	int dimen;
	MLP_layer_train(string path,main_train *mn,int ly):multi_perceptrons_train(path){
		main=mn;
		layer=ly;
		dimen=mn->nervs[layer]->input_dimen;
		mlp_simple_init(dimen,dimen,mn->nervs[layer]->nodes_num,mn->nervs[layer]->type,'l');
		get_compare_data();
		show_compare('c');
		show_compare('t');
		train_init();

		
	}
	void get_compare_data(char mod){
		main->get_compare_data();
		cmp_data &s=(mod=='c')?compare_data:train_cmp_data;
		cmp_data &main_s=(mod=='c')?main->data_set->compare_data:main->data_set->train_cmp_data;
		int num=main_s.num;
		s.init(num,dimen,dimen);
		float *o;
		if(layer==0)o=main_s.trans_input;
		else o=main->layer_out(layer-1,main_s.trans_input,num);
		CUDA_CHECK;
		cudaMemcpy(s.cpu_input,o,sizeof(float)*dimen*num,cudaMemcpyDeviceToHost);
		cudaMemcpy(s.cpu_target,o,sizeof(float)*dimen*num,cudaMemcpyDeviceToHost);
		CUDA_CHECK;
		s.init_data();		
	}
	void get_compare_data(){
		get_compare_data('t');
		get_compare_data('c');
	};
	virtual float show_compare(char mod='c'){
		cmp_data &s=(mod=='c')?compare_data:train_cmp_data;
		set_data_num(s.num);
		s.result=get_result(s.input,s.target);
		coutd<<((mod=='c')?"���ս��":"ѵ�����");
		cout<<" loss:"<<s.result<<" init"<<s.init_result;
		cout<<" rate:"<<(s.result/s.init_result*100)<<"%";
		return s.result;
	}
	virtual void pre_load(int num){
		main->pre_load(num);
	}
	virtual void load_data(){	
		main->load_data();	
		CUDA_CHECK;
		if(layer==0){
			data_input=main->load_train_data_class::input;
			CUDA_CHECK;

		}
		else{
			data_input=main->layer_out(layer-1,main->load_train_data_class::input,main->ctr.data_num);
			CUDA_CHECK;
		}
		
		data_target=data_input;
	}
	void menu(){
		coutd;
		coutd<<"\t2.���³�ʼ��";
		coutd<<"\t3.����ѵ������";
		coutd<<"\t4.������������";
		coutd<<"\t6.������";		
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
			case '6':
				struct_save();
				 multi_perceptrons_train::save_params();
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

class deep_train{
public:
	main_train *main;
	MLP_layer_train *train;
	float *input;
	int layer;
	string root;
	string layer_path;
	string train_path;
	optimization_controller ctr;
	search_set sch;
	deep_train(string p,void *dt){

		root=p;
		layer_path=p+"layer\\";
		file_opt.create_folder(layer_path);
		train=NULL;
		main=new main_train(root,dt,'b');
		if(!ctr.read(layer_path))set_ctr();
		if(!sch.read(layer_path))set_sch();
		run();
	}
	void set_ctr(){
		ctr.set();
		ctr.save(layer_path);
	}
	void set_sch(){
		sch.set();
		sch.save(layer_path);
	}
	struct record{
		char mod;
		int from;
		float dev;
		char reguler_mod;
		float reguler;
		record(){
			reguler_mod='0';
			reguler=0;
			from=0;		
		}
		void load(string path){			
			ifstream fin(path,ios::binary);
			if(!fin)set(path);
			else{
				coutd<<"���ڶ�ȡѵ������"<<path;
				fin.read((char *)this,sizeof(record));				
			}
			show();
		}
		void save(string path){
			cout<<endl<<"���ڱ���ѵ������"<<path;
			ofstream fin(path,ios::binary);
			fin.write((char *)&mod,sizeof(record));		
		}
		void show(){
			coutd<<"\t1.����ģʽ[1:L1 2:L2 0:��ʹ������]:"<<reguler_mod;
			coutd<<"\t2.����ϵ��:"<<reguler;
			coutd<<"\t3.��ʼ����:"<<from;
			coutd<<"\t4.��ֹ���ƫ��:"<<dev;
		}
		void set(string path){
			show();
			coutd<<"\tc.ȷ�� l.�鿴 ";			
			string s;
			do{
				coutd<<"���ò���>";
				cin>>s;
				switch(s[0]){
				case '1':
					cin>>reguler_mod;
					break;
				case '2':
					cin>>reguler;
					break;
				case '3':
					cin>>from;
					break;	
				case '4':
					cin>>dev;
					break;	
				case 'l':
					show();
					coutd<<"\tc.ȷ�� l.�鿴";			
					break;					
				}
			}while(s[0]!='c');
			if(reguler_mod!='1'&&reguler_mod!='2')reguler=0;
			save(path);
		}
	};
	record train_record;
	void run(){
		coutd<<"<���ѵ��>";
		coutd<<"\t1.������������";
		coutd<<"\t2.����ѵ������";
		coutd<<"\t3.������������";
		coutd<<"\tc.��ʼ(����)ѵ�� e.�˳�";
		string s;
		do{
			coutd<<"ѡ��>";
			cin>>s;
			switch(s[0]){
			case '1':
				train_record.set((layer_path+"nerv.rcd"));
				break;
			case '2':
				set_ctr();
				break;
			case '3':
				set_sch();
				break;
			case 'e':
				return;//train_set.set((train_path+"nerv.trn"));
				break;
			}
		}while(s[0]!='c');
		for(int l=train_record.from;l<main->layers_num-1;l++){	
			layer=l;
			layer_train();
		}

	}

	void layer_train(){
		train_record.from=layer;
		train_record.save((layer_path+"nerv.rcd"));
		stringstream s;
		s<<layer_path<<layer<<"\\";
		string path=s.str();
		file_opt.create_folder(path);
		if(!file_opt.check_file(path+ctr.file_name)){
			file_opt.copy(layer_path+ctr.file_name,path+ctr.file_name);
		}
		if(!file_opt.check_file(path+sch.file_name)){
			file_opt.copy(layer_path+sch.file_name,path+sch.file_name);
		}
		//bool flag=file_opt.check_file(layer_path+main->file_name);
		train=new MLP_layer_train(path,main,layer);
		//if(!flag){
			cudaMemcpy(train->weight,main->nervs[layer]->weight,sizeof(float)*main->nervs[layer]->weight_len,cudaMemcpyDeviceToDevice);
			CUDA_CHECK;
		//}
		train->train();
		cudaMemcpy(main->nervs[layer]->weight,train->weight,sizeof(float)*(main->nervs[layer]->weight_len),cudaMemcpyDeviceToDevice);
		CUDA_CHECK;
		main->struct_save();
		coutd<<"�ѽ�ѵ�����������������";
		
		delete train;

	}
};
