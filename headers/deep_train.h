class MLP_simple_train:public multi_perceptrons,public search_tool{
float *input;
float *target;
virtual void load_data()=0;
virtual void pre_load()=0;
virtual  void cacul(){
		cacul_nerv(input,target);	
		ctr.cacul_count++;	
	}
}

class deep_train{
multi_perceptrons_train *main;
multi_perceptrons *train;
float *input;
int layer;
string path;
string train_path;
deep_train(){
		train=NULL;
	}
void load_train_data(int num,char mod='t'){
	 main->pre_load(num,mod);
	 main->load_data();
	 if(layer==1)input=main->load_train_data_class::input);
	 else	 input=layer_out(layer-1,main->load_train_data_class::input,num);
}
void layer_train(int layer,int num){
	int input_dmn=main->nervs[layer]->input_dimen;
	int nodes=main->nervs[layer]->nodes_num;
	train=new multi_perceptrons("");
	train->train_mod=true;
	train->struct_simple_set(input_dmn,input_dmn,nodes,main->nervs[layer]->type,'l');
	train->set_data_num(num);
	set.debug=true;		
	search_init(train->weight_len,&(train->result),train->weight,train->deriv);
	set.set();
	set_init();
	do{
		load_train_data(num);
		search_tool::search(ctr.iteration_num);
	}while(1);
	delete train_mlp;
	sprintf(l_path,"%slayer_%c_%i_%i\\",path.c_str(),train_record.mod,train_record.from,train_record.layer_num);
		coutd<<"当前训练数据路径:"<<l_path;
			train->save_struct();
		train_set.save((train_path+"nerv.trn"));

}
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
			coutd<<"\t1.模式:[ l:逐层训练,n:监督调优,u:无监督调优 ]"<<mod;
			coutd<<"\t2.训练层数:"<<layer_num;
			coutd<<"\t3.起始隐层:"<<from;
			coutd<<"\t4.终止拟合偏差:"<<dev;
		}
		void set(string path){
			show();
			coutd<<"\tc.确定 l.查看 ";			
				string s;
				do{
					coutd<<"设置参数>";
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
						coutd<<"\tc.确定 l.查看";			
						break;					
					}
				}while(s[0]!='c');
			save(path);
		}
	};
	record train_record;
	void layer_train(){
		if(train!=NULL)delete train;
		nerv *nervs=new nerv[train_record.layer_num];
		for(int i=0;i<train_record.layer_num;i++)
			nervs[i]=*(main->nervs[train_record.from-1+i]);
		char l_path[200];
		sprintf(l_path,"%slayer_%c_%i_%i\\",path.c_str(),train_record.mod,train_record.from,train_record.layer_num);
		coutd<<"当前训练数据路径:"<<l_path;
		train=new deep_nerv(l_path,nervs,main,train_record.layer_num);
		train->load_data->deep_train_class=(void *)this;
		train->train_set=train_set;
		train->pre_load_train_data('c');
		train->load_train_data();
		train->start();	
		train->save_struct();
		train_set.save((train_path+"nerv.trn"));
		main->save_struct();
			cout<<endl<<"已将训练结果移入主网络中";
			train_set.save((train_path+"nerv.trn"));
			train_record.from++;
			train_record.save((train_path+"nerv.rcd"));
	
	}
	void run(string p){
		path=p;
		train_path=p+"layer\\";		
		main=new multi_perceptrons_train(path,'r');
		main->create_folder(train_path);
		train_record.load((train_path+"nerv.rcd"));
		train_set.load((train_path+"nerv.trn"));
		cout<<endl<<"<逐层训练>";
		cout<<endl<<"\t1.调整基本设置";
		cout<<endl<<"\t2.调整训练参数";
		cout<<endl<<"\tc.开始(继续)训练 e.退出";
		string s;
		do{
			coutd<<">>";
			cin>>s;
			switch(s[0]){
			case '1':
				train_record.set((train_path+"nerv.rcd"));
				break;
			case '2':
				train_set.set((train_path+"nerv.trn"));
				break;
			case 'e':
				return;//train_set.set((train_path+"nerv.trn"));
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
			
		}

		}
	}
};
