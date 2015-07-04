
struct {
float max_value(float *out,int idx,int dimen){
	float sum=0;
	float maxv=0;
	float tmp;
	for(int i=0;i<dimen;i++){
		tmp=out[idx*dimen+i];
		sum+=tmp;
		if(tmp>maxv)maxv=tmp;
	}
	return maxv/sum;
}

void show_accurercy(string main_path,DATA_CLASS *m){
	main_train p(main_path,m,'r');				
	int num=10000;
	float *ipt=m->input[0];
	float *out=new float[m->output_num*num];
	p.run(ipt,out,m->label[0],num,1,false,false);
	cout<<"\nѵ���� ";
	m->accuracy(out,m->label[0],num);	
	num=m->test_num;
	ipt=m->input_t[0];
	out=new float[m->output_num*num];
	p.run(ipt,out,m->label_t[0],num,1,false,false);
	cout<<"\t���Լ� ";
	m->accuracy(out,m->label_t[0],num);
	delete [] out;
}
void show_wrong_sample(string main_path,DATA_CLASS *m){
	main_train p(main_path,m,'r');				
	int num=m->test_num;
	float *ipt=m->input_t[0];
	float *out=new float[m->output_num*num];
	p.run(ipt,out,m->label_t[0],num,1,false,false);
	coutd<<"���Լ�ʶ������������:";
	coutd;
	for(int i=0;i<num;i++){
			int v=m->value(m->label_t[0]+i*m->output_num);
			int o=m->get_out_value(out+i*m->output_num);
			if(v!=o){
				cout<<i<<"\t";
			}
	}
	coutd;
	do{
			int idx=m->input_idx('t');
			if(idx==-1)break;
			if(idx==-2)continue;
			m->show(idx,'t');
			coutd<<"ʶ����:"<<(m->get_out_value(out+idx*m->output_num));
		}while(1);
}
void check_result(string main_path,DATA_CLASS *m){
	string sel;
	do{
		coutd;
		coutd<<"\ta.�鿴��ȷ��";
		coutd<<"\tb.�鿴ʶ����������";
		coutd<<"\te.����";
		coutd<<"ѡ��>";
		cin>>sel;
		switch(sel[0]){
		case 'a':
			show_accurercy(main_path,m);
			break;
		case 'b':
			show_wrong_sample(main_path,m);
			break;
		case 'e':
			return;
		}
	}while(1);
}

string main_path;
string data_path;
void check_distortion(){
	DATA_CLASS *m=new DATA_CLASS(data_path,main_path,true);
	float *map;
	int rows=m->rows;
	int columns=m->columns;
	cudaMalloc((void **)&map,sizeof(float)*rows*columns*1000);	
	float *c_map=new float[rows*columns*1000];
	memcpy(c_map,m->data,sizeof(float)*rows*columns*1000);		
	do{	
		cudaMemcpy(map,c_map,sizeof(float)*rows*columns*1000,cudaMemcpyHostToDevice);	
		memcpy(m->data,c_map,sizeof(float)*rows*columns*1000);	
		char mod;
		coutd;
		coutd<<"\t<�鿴distortionЧ��>";
		coutd<<"\te.����Ť��";
		coutd<<"\tm.ƽ�Ʊ任";
		coutd<<"\ts.���ű任";
		coutd<<"\tr.��ת�任";
		coutd<<"\tt.���б任";
		coutd<<"\tb.��Ϸ���Ť��";
		coutd<<"\ta.����+����Ť��";
		coutd<<"\t��l.�鿴Ť������ k.�޸�Ť������ c.���أ� ";
		coutd<<"ѡ��Ť��ģʽ>";
		cin>>mod;
		if(mod=='c')break;
		if(mod=='l'){
			m->eldt->d_set.show();
			continue;
		}
		if(mod=='k'){
			m->eldt->set();
			continue;
		}
		m->eldt->mod_distortion(map,1000,mod);
		for(int i=0;i<1000;i++){
			coutd<<"<ԭʼͼ��>";
			m->show(i,'i',false);
			cudaMemcpy(m->data+i*rows*columns,map+i*rows*columns,sizeof(float)*rows*columns,cudaMemcpyDeviceToHost);
			CUDA_CHECK;
			coutd<<"<distortionͼ��>";
			m->show(i,'i',false);
			coutd<<"(�س�������������e���أ�";
			char c=getchar();			
			if(c=='e')break;
		}					
	}while(1);	
	delete m;
	cudaFree(map);
	delete [] c_map;
}
void show_bagging_result(DATA_CLASS *m,string main_path){
	bagging p(main_path,m,false);
	cout<<"\n����MLP��ȷ��:";
	cout_show=false;
	for(int i=0;i<p.nervs_num;i++){
		cout<<"\n"<<p.get_idx_name(i);
	show_accurercy(p.get_idx_name(i),m);
	}
	cout_show=true;
	coutd<<"\n\nbagging���:";
	int num=10000;
	float *ipt=m->input[0];
	float *out=new float[m->output_num*num];
	p.run(ipt,out,m->label[0],num,1,false,false);
	cout<<"\nѵ���� ";
	m->accuracy(out,m->label[0],num);
	delete [] out;
	num=m->test_num;
	ipt=m->input_t[0];
	out=new float[m->output_num*num];
	p.run(ipt,out,m->label_t[0],num,1,false,false);
	cout<<"\t���Լ� ";
	m->accuracy(out,m->label_t[0],num);
	

}
void start(){
		
			coutd<<"***********************************************************************";
			coutd;
			coutd<<"                           MNIST TOY 1.1";
			coutd;
			coutd<<"                         ����:�������� ";
			coutd;
			coutd<<"***********************************************************************";
			g_gpu_init();
			main_path="";//"f:\\mnist\\";//[EDIT_SYMBOL]
			data_path="data_set\\";//"f:\\mnist\\data_set\\";//[EDIT_SYMBOL]
			string name;
			coutd;
		do{
			coutd<<"������Ŀ����:('-'��ʾ������Ŀ��";
			cin>>name;
			if(name[0]=='-'){
				file_opt.show_project_name(main_path,image_distortion_set_file);
				continue;
			}
			string tmp=main_path+name+"\\";
			if(file_opt.check_folder(tmp)){
				main_path=tmp;
				break;
			}
			coutd<<"�����ڴ���Ŀ���Ƿ�Ҫ����?(Y/N)";
			string sel;
			cin>>sel;
			if(sel[0]=='y'||sel[0]=='Y'){
				main_path=tmp;
				file_opt.create_folder(main_path);
				break;
			}
		}while(1);
	}

void menu(){
			coutd;	
			coutd<<"\t1.�鿴 mnist ͼ��";
			coutd<<"\t2.�鿴 distortion Ч��";			
			coutd<<"\t3.ѵ��";
			coutd<<"\t4.�鿴ѵ�����";
			coutd<<"\t5.baggingѵ��";
			coutd<<"\t6.bagging����";
			coutd<<"ѡ��>";
			char sel;
			cin>>sel;
			switch(sel){
				case '1':{
					mnist *m=new mnist(data_path);
					m->check();
					delete [] m;
					}
					break;
				case '2':
					check_distortion();
					break;
				case '3':{	
					DATA_CLASS *m=new DATA_CLASS(data_path,main_path);
					main_train s(main_path,m);
					delete m;
					break;
					   }
				case '4':{
					g_gpu_init();
					DATA_CLASS *m=new DATA_CLASS(data_path,main_path,true);
					check_result(main_path,m);
						 }
					break;	
				case '5':{
					DATA_CLASS *m=new DATA_CLASS(data_path,main_path);
					bagging tt(main_path,m,true);
						 }
					break;
				case '6':{
					DATA_CLASS *m=new DATA_CLASS(data_path,main_path,true);
					show_bagging_result(m,main_path);
					
						 }
					break;
				
			}
		
}

void main(){
	start();
	do{
		menu();
	}while(1);	
	
}
}mnist_main;


