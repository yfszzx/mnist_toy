
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
	multi_perceptrons_train p(main_path,m,'r');				
	int num=10000;
	float *ipt=m->input[0];
	float *out=new float[m->output_num*num];
	p.run(ipt,out,m->label[0],num,1,-1,false,false);
	cout<<"\n训练集 ";
	m->accuracy(out,m->label[0],num);	
	num=m->test_num;
	ipt=m->input_t[0];
	out=new float[m->output_num*num];
	p.run(ipt,out,m->label_t[0],num,1,-1,false,false);
	cout<<"\n测试集 ";
	m->accuracy(out,m->label_t[0],num);
	delete [] out;
}
void show_wrong_sample(string main_path,DATA_CLASS *m){
	multi_perceptrons_train p(main_path,m,'r');				
	int num=m->test_num;
	float *ipt=m->input_t[0];
	float *out=new float[m->output_num*num];
	p.run(ipt,out,m->label_t[0],num,1,-1,false,false);
	coutd<<"测试集识别错误样本序号:";
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
			coutd<<"识别结果:"<<(m->get_out_value(out+idx*m->output_num));
		}while(1);
}
void check_result(string main_path,DATA_CLASS *m){
	string sel;
	do{
		coutd;
		coutd<<"\ta.查看正确率";
		coutd<<"\tb.查看识别错误的样本";
		coutd<<"\te.返回";
		coutd<<"选择>";
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
		coutd<<"\t<查看distortion效果>";
		coutd<<"\te.弹性扭曲";
		coutd<<"\tm.平移变换";
		coutd<<"\ts.缩放变换";
		coutd<<"\tr.旋转变换";
		coutd<<"\tt.剪切变换";
		coutd<<"\tb.组合仿射扭曲";
		coutd<<"\ta.仿射+弹性扭曲";
		coutd<<"\t（l.查看扭曲参数 k.修改扭曲参数 c.返回） ";
		coutd<<"选择扭曲模式>";
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
			coutd<<"<原始图像>";
			m->show(i,'i',false);
			cudaMemcpy(m->data+i*rows*columns,map+i*rows*columns,sizeof(float)*rows*columns,cudaMemcpyDeviceToHost);
			CUDA_CHECK;
			coutd<<"<distortion图像>";
			m->show(i,'i',false);
			coutd<<"(回车键继续，输入e返回）";
			char c=getchar();			
			if(c=='e')break;
		}					
	}while(1);	
	delete m;
	cudaFree(map);
	delete [] c_map;
}
void start(){
		
			coutd<<"***********************************************************************";
			coutd;
			coutd<<"                           MNIST TOY 1.0";
			coutd;
			coutd<<"                         作者:大隐于市 ";
			coutd;
			coutd<<"***********************************************************************";
			g_gpu_init();
			main_path="";//"f:\\mnist\\";
			data_path="data_set\\";//"f:\\mnist\\data_set\\";
			string name;
			coutd;
		do{
			coutd<<"输入项目名称:('-'显示所有项目）";
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
			coutd<<"不存在此项目，是否要创建?(Y/N)";
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
			coutd<<"\t1.查看 mnist 图像";
			coutd<<"\t2.查看 distortion 效果";			
			coutd<<"\t3.训练";
			coutd<<"\t4.查看训练结果";
			//coutd<<"\t5.bagging训练";
			//coutd<<"\t6.bagging检验";
			coutd<<"选择>";
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
					multi_perceptrons_train s(main_path,m);
					delete m;
					break;
					   }
				case '4':{
					g_gpu_init();
					DATA_CLASS *m=new DATA_CLASS(data_path,main_path,true);
					check_result(main_path,m);
						 }
					break;	
			/*	case '5':{
					DATA_CLASS *m=new DATA_CLASS(data_path,main_path);
					bagging tt(main_path,m,true);
						 }
					break;
				case '6':{
					DATA_CLASS *m=new DATA_CLASS(data_path,main_path,true);
					coutd<<"输入bagging文件夹名(默认为'-'):";
					string nm;
					cin>>nm;
					if(nm[0]!='-')main_path+=nm+"\\";
					bagging p(main_path,m,false);
					coutd<<"输入投票方式:";
					char v;
					cin>>v;
					p.vote_mod=v;
					int num=10000;
					float *ipt=m->input[0];
					float *out=new float[m->output_num*num];
					cout_show=false;
					p.run(ipt,out,m->label[0],num);
					cout<<"\n训练集 ";
					cout<<"正确率:"<<m->accuracy(out,m->label[0],num);
					delete [] out;
					num=m->test_num;
					ipt=m->input_t[0];
					out=new float[m->output_num*num];
					p.run(ipt,out,m->label_t[0],num);
					cout<<"\t测试集 ";
					cout<<"正确率:"<<m->accuracy(out,m->label_t[0],num);
					cout<<"\n各个bagging正确率:";
					for(int i=0;i<p.nervs_num;i++){
						show_accurercy(p.get_idx_name(i),m);
					}
					getchar();getchar();
						 }
					break;
				*/
			}
		
}

void main(){
	start();
	do{
		menu();
	}while(1);	
	
}
}mnist_main;


