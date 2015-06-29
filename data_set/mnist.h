/*
mnist资料:
http://yann.lecun.com/exdb/mnist/

elastic distortion 参考文献
http://www.tuicool.com/articles/IVzUFbI
 注意:此文档的高斯函数有误

Best Pracies forCnvlutioa Neurltwoks Aplied toVsual Docment Aalysis
http://research.microsoft.com/pubs/68920/icdar03.pdf

 */
string image_distortion_set_file="distortion_set.stl";
__global__ void g_affine(float *mtx,float *rnd_x,float *rnd_y,float scl,float r,char mod,int num,bool init){	
	int idx=threadIdx.x;
	int img_idx=blockDim.y*blockIdx.x+threadIdx.y;
	int pos_idx=img_idx*6+idx;
	int a_pos=threadIdx.y*6;
	int m_pos=blockDim.y*6+a_pos;
	int s_pos=blockDim.y*6+m_pos;
	__shared__ extern float af_mtx[];
	if(img_idx<num)af_mtx[a_pos+idx]=0;
	__syncthreads();
	if(img_idx<num){
	if(idx==0){
		float x,y;
		switch(mod){
		case 'm'://move
			x=(rnd_x[img_idx]*2-1)*scl;
			y=(rnd_y[img_idx]*2-1)*scl;
			af_mtx[a_pos+0]=1;
			af_mtx[a_pos+2]=x;
			af_mtx[a_pos+4]=1;
			af_mtx[a_pos+5]=y;
			break;
		case 's'://scale
			x=(rnd_x[img_idx]*2-1)*scl;
			y=(rnd_y[img_idx]*2-1)*scl;
			af_mtx[a_pos+0]=(x+r*2)/r/2;
			af_mtx[a_pos+4]=(y+r*2)/r/2;
			break;
		case 't'://shear
			x=(rnd_x[img_idx]*2-1)*scl;
			y=(rnd_y[img_idx]*2-1)*scl;
			af_mtx[a_pos+0]=1;
			af_mtx[a_pos+1]=x/r;
			af_mtx[a_pos+4]=1;
			af_mtx[a_pos+3]=y/r;
			break;
		case 'r'://rotate
			x=(rnd_x[img_idx]*2-1)*scl;
			float arg=x/r;	
			af_mtx[a_pos+0]=cos(arg);
			af_mtx[a_pos+1]=-sin(arg);
			af_mtx[a_pos+4]=cos(arg);
			af_mtx[a_pos+3]=sin(arg);
			break;
		}
	}
	if(init)
		af_mtx[m_pos+idx]=(idx==0||idx==4)?1.0f:0;
	else
		af_mtx[m_pos+idx]=mtx[pos_idx];
	}
	__syncthreads();
	if(img_idx<num&&idx==0){
		af_mtx[s_pos+0]=af_mtx[a_pos+0]*af_mtx[m_pos+0]+af_mtx[a_pos+1]*af_mtx[m_pos+3];
		af_mtx[s_pos+1]=af_mtx[a_pos+0]*af_mtx[m_pos+1]+af_mtx[a_pos+1]*af_mtx[m_pos+4];
		af_mtx[s_pos+2]=af_mtx[a_pos+0]*af_mtx[m_pos+2]+af_mtx[a_pos+1]*af_mtx[m_pos+5]+af_mtx[a_pos+2];
		af_mtx[s_pos+3]=af_mtx[a_pos+3]*af_mtx[m_pos+0]+af_mtx[a_pos+4]*af_mtx[m_pos+3];
		af_mtx[s_pos+4]=af_mtx[a_pos+3]*af_mtx[m_pos+1]+af_mtx[a_pos+4]*af_mtx[m_pos+4];
		af_mtx[s_pos+5]=af_mtx[a_pos+3]*af_mtx[m_pos+2]+af_mtx[a_pos+4]*af_mtx[m_pos+5]+af_mtx[a_pos+5];
	}
	__syncthreads();
	if(img_idx<num)mtx[pos_idx]=af_mtx[s_pos+idx];
}
__global__ void g_affine_transform(float *map_x,float *map_y,float *aff_mtx){
	int r=blockDim.y;
	int c=blockDim.x;
	int x=threadIdx.x;
	int y=threadIdx.y;
	int idx=y*c+x;
	float c_x=c/2;
	float c_y=r/2;
	int pos_idx=blockIdx.x*r*c+idx;
	__shared__ float mtx[6];
	if(idx<6)mtx[x]=aff_mtx[blockIdx.x*6+idx];
	__syncthreads();
	map_x[pos_idx]=mtx[0]*(x-c_x)-mtx[1]*(y-c_y)+mtx[2]+c_x;
	map_y[pos_idx]=mtx[3]*(x-c_x)+mtx[4]*(y-c_y)+mtx[5]+c_y;
	}
__global__ void g_elasitc(float *img_x,float *img_y,float intensive,float dev_r,int kernel_scl,float knl_uniform){	
	int r=blockDim.y;
	int c=blockDim.x;
	int x=threadIdx.x;
	int y=threadIdx.y;
	int idx=c*y+x;
	int pos_idx=c*r*blockIdx.x+idx;	
	int ks2= kernel_scl* kernel_scl;
	__shared__ extern float knl[];
	int cnt=(kernel_scl+1)/2-1;
	if(x<kernel_scl&&y<kernel_scl)
		knl[y*kernel_scl+x]=__expf(-float((x-cnt)*(x-cnt)+(y-cnt)*(y-cnt))/dev_r)/knl_uniform;
	knl[ks2+idx]=img_x[pos_idx]*2-1;
	knl[ks2+c*r+idx]=img_y[pos_idx]*2-1;
	__syncthreads();
	float sum_x=0;
	float sum_y=0;
	int yy,xx;	
	for(int i=0;i<kernel_scl;i++){
		yy=y-cnt+i;
		if(yy<0||yy>=r)continue;
		for(int j=0;j<kernel_scl;j++){
			xx=x-cnt+j;
			if(xx<0||xx>=c)continue;
			sum_x+=knl[ks2+yy*c+xx]*knl[i*kernel_scl+j];
			sum_y+=knl[ks2+c*r+yy*c+xx]*knl[i*kernel_scl+j];
		}
	}
	img_x[pos_idx]=sum_x*intensive+x;
	img_y[pos_idx]=sum_y*intensive+y;
}
__global__ void g_distortion(float *imgs,float *map_x,float *map_y){
	int r=blockDim.y;
	int c=blockDim.x;
	int x=threadIdx.x;
	int y=threadIdx.y;
	int idx=c*y+x;
	int pos_idx=c*r*blockIdx.x+idx;
	__shared__ extern float img[];
	img[idx]=imgs[pos_idx];
	__syncthreads();
	float xd=map_x[pos_idx];
	float yd=map_y[pos_idx];
	if(xd<0||xd>=c-1||yd<0||yd>=r-1){
		imgs[pos_idx]=0;
		return;
	}
	int xp=int(xd);
	int yp=int(yd);
	xd-=xp;
	yd-=yp;
	imgs[pos_idx]=img[yp*c+xp]*(1-xd)*(1-yd)+img[(yp+1)*c+xp]*yd*(1-xd)+img[yp*c+xp+1]*(1-yd)*xd+img[(yp+1)*c+xp+1]*xd*yd;
};
class image_distortion{
public:
	float *map_x;
	float *map_y;
	float *rnd_x;
	float *rnd_y;
	float *aff_mtx;
	float r;
	int knl_scl;
	float knl_uniform;//归一化kernel矩阵
	int data_num;
	int rows;
	int columns;
	curandGenerator_t gen;
	string path;
	image_distortion(int rr,int cc,string root){
		path=root+image_distortion_set_file;
		read();
		map_x=NULL;
		map_y=NULL;
		data_num=0;
		rows=rr;
		columns=cc;
		r=(rr+cc)/4;
		set_kernel_scl();		
		CHECK_CURAND( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
		CHECK_CURAND( curandSetPseudoRandomGeneratorSeed(gen,time(NULL)+rand()));
	}
	~image_distortion(){
		memfree();
		CHECK_CURAND( curandDestroyGenerator(gen) );
	}
	void set_kernel_scl(){
		knl_scl=0;
		for(int i=1;i<(rows-1)/2;i++){//边缘的kernel值应小于中心值的1/10
			if(1.0f/expf(-float(i*i)/(2.0f*d_set.dev*d_set.dev))>10.0f){
				knl_scl=i*2+1;
				break;
			}
		}
		if(knl_scl==0)knl_scl=(rows%2==1)?rows:(rows-1);
		int x,y;
		int cnt=(knl_scl+1)/2-1;
		knl_uniform=0;
		for(int i=0;i<knl_scl;i++){
			x=i-cnt;
			for(int j=0;j<knl_scl;j++){
				y=j-cnt;
				knl_uniform+=expf(-float(x*x+y*y)/(2.0f*d_set.dev*d_set.dev));
			}
		}
	//coutd<<"kernel宽度:"<<knl_scl<<" kernel中心值:"<<(1.0f/knl_uniform);

	}
	void memfree(){
		if(map_x!=NULL){
			cudaFree(map_x);
			cudaFree(map_y);
			cudaFree(rnd_x);
			cudaFree(rnd_y);
			cudaFree(aff_mtx);
			CUDA_CHECK;
		}
	}
	void set_data_num(int num){
		if(num==data_num)return;
		memfree();
		data_num=num;
		cudaMalloc((void **)&map_x,sizeof(float)*(rows*columns*data_num+1));
		cudaMalloc((void **)&map_y,sizeof(float)*(rows*columns*data_num+1));
		cudaMalloc((void **)&rnd_x,sizeof(float)*(data_num+1));
		cudaMalloc((void **)&rnd_y,sizeof(float)*(data_num+1));
		cudaMalloc((void **)&aff_mtx,sizeof(float)*6*data_num);
		CUDA_CHECK;
	}
	void aff_rand_map(bool both=true){
		int num=data_num;
		num=(num%2==1)?(num+1):num;
		CHECK_CURAND( curandGenerateUniform(gen, rnd_x,num));
		if(both)CHECK_CURAND( curandGenerateUniform(gen, rnd_y,num));
	}
	void ela_rand_map(){	
		int num=data_num*rows*columns;
		num=(num%2==1)?(num+1):num;
		CHECK_CURAND( curandGenerateUniform(gen,map_x,num) );		
		CHECK_CURAND( curandGenerateUniform(gen,map_y,num) );		
		CUDA_CHECK;
	}
	void mod_distortion(float *imgs,int num,char mod){
		set_data_num(num);
		dim3 thd(columns,rows);
		if(mod!='b'&&mod!='e'&&mod!='a'){
		aff_rand_map();
		int thdn=g_threads/6;
		int blk=(data_num+thdn-1)/thdn;
		dim3 thdd(6,thdn);
		g_affine<<<blk,thdd,sizeof(float)*thdn*6*3>>>(aff_mtx,rnd_x,rnd_y,d_set.affine_scale,r,mod,data_num,true);
		CUDA_CHECK;
		g_affine_transform<<<data_num,thd>>>(map_x,map_y,aff_mtx);	
		CUDA_CHECK;
		g_distortion<<<data_num,thd,sizeof(float)*columns*rows>>>(imgs,map_x,map_y);
		CUDA_CHECK;
		}
		if(mod=='e')ela_distortion(imgs,num);
		if(mod=='b')aff_distortion(imgs,num);
		if(mod=='a'){
			aff_distortion(imgs,num);
			ela_distortion(imgs,num);
		}
	}
	void aff_distortion(float *imgs,int num){
		if(d_set.affine_scale==0)return;
		set_data_num(num);
		
		int thdn=g_threads/6;
		int blk=(data_num+thdn-1)/thdn;
		dim3 thdd(6,thdn);
		
		aff_rand_map(false);
		g_affine<<<blk,thdd,sizeof(float)*thdn*6*3>>>(aff_mtx,rnd_x,rnd_y,d_set.affine_scale*d_set.rotate,r,'r',data_num,true);
		CUDA_CHECK;
		aff_rand_map(true);
		g_affine<<<blk,thdd,sizeof(float)*thdn*6*3>>>(aff_mtx,rnd_x,rnd_y,d_set.affine_scale*d_set.shear,r,'t',data_num,false);
		CUDA_CHECK;
		aff_rand_map(true);
		g_affine<<<blk,thdd,sizeof(float)*thdn*6*3>>>(aff_mtx,rnd_x,rnd_y,d_set.affine_scale*d_set.move,r,'m',data_num,false);
		CUDA_CHECK;			
		aff_rand_map(true);
		g_affine<<<blk,thdd,sizeof(float)*thdn*6*3>>>(aff_mtx,rnd_x,rnd_y,d_set.affine_scale*d_set.scale,r,'s',data_num,false);
		CUDA_CHECK;
		dim3 thd(columns,rows);
		g_affine_transform<<<data_num,thd>>>(map_x,map_y,aff_mtx);
		CUDA_CHECK;
		g_distortion<<<data_num,thd,sizeof(float)*columns*rows>>>(imgs,map_x,map_y);
		CUDA_CHECK;

	}
	void ela_distortion(float *imgs,int num){
		if(d_set.intensive==0)return;
		set_data_num(num);
		ela_rand_map();
		dim3 thd(columns,rows);
		g_elasitc<<<data_num,thd,sizeof(float)*columns*rows*3>>>(map_x,map_y,d_set.intensive,2.0f*d_set.dev*d_set.dev,knl_scl,knl_uniform);
		CUDA_CHECK;
		g_distortion<<<data_num,thd,sizeof(float)*columns*rows>>>(imgs,map_x,map_y);
		CUDA_CHECK;
	}
	struct distortion_set{
		float dev;
		float intensive;
		float move;
		float scale;
		float rotate;
		float shear;
		float affine_scale;
		distortion_set(){
			dev=5;
			intensive=38;
			move=1;
			scale=1;
			rotate=1;
			shear=1;
			affine_scale=0;
		}
		void show(){
			coutd;
			coutd<<"\t<distorion参数>";
			coutd<<"\t\td.[弹性]高斯函数标准差:"<<dev;
			coutd<<"\t\ti.[弹性]敏感系数(intensity):"<<intensive;
			coutd<<"\t\ta.[仿射]形变尺度:"<<affine_scale;
			coutd<<"\t\tm.[组合]平移比例:"<<move;
			coutd<<"\t\ts.[组合]缩放比例:"<<scale;
			coutd<<"\t\tr.[组合]旋转比例:"<<rotate;
			coutd<<"\t\tt.[组合]剪切比例:"<<shear;

		}
		void set(){
			show();
			coutd<<"\tc.确定 l.查看";
			coutd;
			coutd<<"·取消弹性扭曲，将\"i.[弹性]敏感系数\"设为0";
			coutd<<"·取消仿射扭曲，将\"a.[仿射]形变尺度\"设为0";
			coutd;
	
			do{
				coutd<<"选择参数>";
				string sel;
				cin>>sel;
				switch(sel[0]){
				case 'd':
					cin>>dev;
					break;
				case 'i':
					cin>>intensive;
					break;
				case 'a':
					cin>>affine_scale;
					break;
				case 'm':
					cin>>move;
					break;
				case 's':
					cin>>scale;
					break;
				case 'r':
					cin>>rotate;
					break;
				case 't':
					cin>>shear;
					break;
				case 'l':
					show();
					break;
				case 'c':
					return;
					break;
				}				
			}while(1);
		
		}
	};
		
	distortion_set d_set;
	void set(){
		d_set.set();
		set_kernel_scl();
		save();
	}
	void read(){
			if(!file_opt.check_file(path)){
				set();
				return;
			}
		ifstream fin(path,ios::binary);
		coutd<<"正在读取distortion参数"<<path;
		fin.read((char *)&d_set,sizeof(distortion_set));		
	}
	void save(){
		ofstream fin(path,ios::binary);
		coutd<<"正在保存distortion参数"<<path;
		fin.write((char *)&d_set,sizeof(distortion_set));		
	}


};
		
		


class mnist{
public:
	float *data;
	float *label_d;
	float *data_t;
	float *label_t_d;
	float **label;
	float **label_t;
	float **input;
	float **input_t;
	int input_num;
	int output_num;
	int rows;
	int columns;
	int train_num;
	int test_num;
	mnist(string path){
		
		int t;
		unsigned char *tmp;
		coutd<<"正在载入数据集....";
		//数据
		ifstream fin(path+"train-images.idx3-ubyte", ios::binary); 	
		if(!fin){
			cout<<"路径错误";
			char s;
			cin>>s;
			return;
		}
	
		fin.read((char *)&t,sizeof(int));//magic_num;
		fin.read((char *)&train_num,sizeof(int));//data_num		
		fin.read((char *)&rows,sizeof(int));
		fin.read((char *)&columns,sizeof(int));
		rows=28;
		columns=28;
		train_num=60000;
		input_num=rows*columns;
		output_num=10;
		coutd<<"行数:"<<rows<<" 列数:"<<columns<<endl<<"训练集数量:"<<train_num;
		tmp=new unsigned char[rows*columns*train_num*sizeof(char)];
		fin.read((char *)tmp,rows*columns*train_num*sizeof(char));
		
		data=new float[rows*columns*train_num*sizeof(float)];
		for(int i=0;i<rows*columns*train_num;i++){
		
			data[i]=float(tmp[i])/255;
		}
		input=new float *[train_num];
		for(int i=0;i<train_num;i++)
			input[i]=data+rows*columns*i;
		delete [] tmp;
		fin.close();
		//标签
		fin.open(path+"train-labels.idx1-ubyte",ios::binary); 	
		if(!fin){
			cout<<"路径错误";
			char s;
			cin>>s;
			return;
		}
		fin.read((char *)&t,sizeof(int));//magic_num;
		fin.read((char *)&t,sizeof(int));//data_num	
		tmp=new unsigned char[train_num*sizeof(char)];
		fin.read((char *)tmp,train_num*sizeof(char));
		label_d=new float[train_num*output_num*sizeof(float)];		
		label=new float *[train_num*sizeof(float)];
		for(int i=0;i<train_num;i++){			
			label[i]=label_d+i*output_num;
			for(int j=0;j<output_num;j++){
				label[i][j]=(j==tmp[i])?1:0;
			}
			
		}
		delete [] tmp;	
		fin.close();
				
		
		
		//测试集
		//数据
		fin.open(path+"t10k-images.idx3-ubyte", ios::binary); 		
		fin.read((char *)&t,sizeof(int));//magic_num;
		fin.read((char *)&test_num,sizeof(int));//data_num
		fin.read((char *)&t,sizeof(int));
		fin.read((char *)&t,sizeof(int));
		test_num=10000;
		cout<<"测试集数量:"<<test_num;
		tmp=new unsigned char[rows*columns*test_num*sizeof(char)];
		fin.read((char *)tmp,rows*columns*test_num*sizeof(char));		
		data_t=new float[rows*columns*test_num*sizeof(float)];
		for(int i=0;i<rows*columns*test_num;i++)
			data_t[i]=float(tmp[i])/255;
		input_t=new float *[test_num];
		for(int i=0;i<test_num;i++)
			input_t[i]=data_t+rows*columns*i;
		delete [] tmp;
		fin.close();
		//标签
		fin.open(path+"t10k-labels.idx1-ubyte", ios::binary); 		
		fin.read((char *)&t,sizeof(int));//magic_num;
		fin.read((char *)&t,sizeof(int));//data_num		
		tmp=new unsigned char[test_num*sizeof(char)];
		fin.read((char *)tmp,test_num*sizeof(char));
		label_t_d=new float[test_num*output_num*sizeof(float)];		
		label_t=new float *[test_num*sizeof(float)];
		for(int i=0;i<test_num;i++){			
			label_t[i]=label_t_d+i*output_num;
			for(int j=0;j<output_num;j++){
				label_t[i][j]=(j==tmp[i])?1:0;
			}
		}
		delete [] tmp;
		fin.close();
	}

	void show(int idx,char type,bool label_show=true){
		float **&ipt=(type=='i')?input:input_t;
		//float **&lbl=(type=='i')?label:label_t;
		coutd;
		for(int j=0;j<columns+2;j++)cout<<"-";
		coutd;
		for(int i=0;i<rows;i++){
			cout<<"|";
			for(int j=0;j<columns;j++){
				if(ipt[idx][i*rows+j]<0.25){
					cout<<" ";
					continue;
				}
				if(ipt[idx][i*rows+j]<0.5){
					cout<<".";
					continue;
				}
				if(ipt[idx][i*rows+j]<0.75){
					cout<<"+";
					continue;
				}
				cout<<"*";
			}
			cout<<"|\n";
		}		
		for(int j=0;j<columns+2;j++)cout<<"-";
		if(label_show)coutd<<"序号:"<<idx<<" 数字:"<<value(idx,type);
		cout<<endl;
	}	
	int input_idx(char type){
		int idx;
		coutd<<"输入序号(-1返回):";
			cin>>idx;
			if(idx<0)return -1;
			if(type=='i'){
				if(idx>=train_num){
					coutd<<"【错误】序号太大";
					return -2;
				}
			}else{
				if(idx>=test_num){
					coutd<<"【错误】序号太大";
					return -2;
				}
			}
		return idx;
	}
	void check(){
		int idx;
		char type;
		coutd;
		coutd<<"\t<查看mnist图像>";
		coutd<<"选择数据集[i.训练集 t.测试集]:";
		cin>>type;
		do{
			idx=input_idx(type);
			if(idx==-1)break;
			if(idx==-2)continue;
			show(idx,type);
		}while(1);
	}
	int value(float *tgt){
		for(int i=0;i<output_num;i++)	if(tgt[i]>0)return i;
		return -1;
	}
	int value(int idx,char mod){
		if(mod=='i')return value(label[idx]);
		else return value(label_t[idx]);
	}
	int get_out_value(float *out){
		float mx=-10;int ret=-1;
		for(int i=0;i<output_num;i++){
			if(out[i]>mx){
				mx=out[i];
				ret=i;
			}
		}
		return ret;
	}
	void show_out(float *out){
		cout<<"result:";
		int val=get_out_value(out);
		cout<<val<<endl;		
		for(int i=0;i<output_num;i++)
			cout<<" "<<i<<":"<<out[i];
	}
	float accuracy(float *out,float *tgt,int num){
		int count=0;	
		for(int i=0;i<num;i++){
			int v=value(tgt+i*output_num);
			int o=get_out_value(out+i*output_num);
			if(v==o)count++;			
		}
		float ret=float(count)/num;
		cout<<" 正确率:"<<(ret*100)<<"%("<<num<<"个样本)";
		return ret;
	}
	void show_wrong(float *out,char mod='c'){
		int count=0;
		int num=(mod=='t')?train_num:test_num;
		for(int i=0;i<num;i++){
			int v=value(i,mod);
			int o=get_out_value(out+output_num*i);
			if(v==o)count++;
			else{
				show(i,mod);
				show_out(out+output_num*i);
				getchar();
			}
		}
	
	}
};
class mnist_data_set:public virtual_data_set,public mnist{
public:
	image_distortion *eldt;
	mnist_data_set(string root,string path,bool default_param=false):mnist(root){
		input_dimen=input_num;
		output_dimen=output_num;
		action_illu="设置distortion参数";
		sample_scale_set(0,train_num,0,test_num);
		eldt=new image_distortion(rows,columns,path);
	}
	virtual void get_data(char mod,float *in,float *out,int num){
	float **ipt=(mod=='t'||mod=='p'||bag_mod)?input:input_t;
	float **lbl=(mod=='t'||mod=='p'||bag_mod)?label:label_t;
	for(int i=0;i<num;i++){
		int idx;
		idx=rand_idx(mod);
		if(!bag_mod&&mod=='c')idx=i%test_num;
		memcpy(in+i*input_num,ipt[idx],sizeof(float)*input_num);
		memcpy(out+i*output_num,lbl[idx],sizeof(float)*output_num);
	}
}
	virtual void show_compare(char mod='c'){	
		cmp_data &s=(mod=='c')?compare_data:train_cmp_data;
		coutd<<((mod=='c')?"对照结果":"训练结果");
		cout<<" loss:"<<s.result<<" ";
		accuracy(s.cpu_output,s.cpu_target,s.num);
	};
	virtual void self_action(void *mlp){
		eldt->set();
	};
	virtual void gpu_distortion(float *input,int num){
		eldt->aff_distortion(input,num);
		eldt->ela_distortion(input,num);
	};
};

