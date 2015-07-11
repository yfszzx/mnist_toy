
class array_operate{
public:
	int dimen;

	struct saxpy_functor{
		const float a;
		saxpy_functor ( float _a) : a(_a) {}
		__host__ __device__	float operator ()( const float & x, const float & y) const {
			return a * x + y;
		}
	};

	void add(float *arr,float *arr1,float *arr2,float x){//arr=arr1*x+arr2
		thrust::device_ptr<float >  a1 ( arr1 );
		thrust::device_ptr<float >  a2 ( arr2 );
		thrust::device_ptr<float >  a ( arr );
		thrust::transform (a1 , a1+dimen , a2 , a , saxpy_functor (x));
	}	
	void simple_add(float *arr1,float *arr2,float x){//arr1=arr1+arr2*x ,此函数比add()有更快的速度
		CUBT(cublasSaxpy (cublasHandle,dimen,&x,arr2,1, arr1,1));
	}
	float dot(float *arr1,float *arr2){//矢量点积
		float ret;
		CUBT(cublasSdot (cublasHandle,dimen, arr1,1, arr2, 1,&ret));		
		return ret;
	}	

	void zero(float *arr){
		cudaMemset(arr,0,sizeof(float)*dimen);
		CUDA_CHECK;
	}
	void scale(float *arr,float x){//arr=arr*x;
		CUBT(cublasSscal(cublasHandle,dimen, &x, arr, 1));
	}
	void clone(float *arr_from,float *arr_to){
		cudaMemcpy(arr_to,arr_from,sizeof(float)*dimen,cudaMemcpyDeviceToDevice);
		CUDA_CHECK;
	}
	float length(float *arr){
		float ret;
		CUBT(cublasSnrm2 (cublasHandle,dimen, arr, 1,&ret));
		return ret;
	}
};
struct search_set{
	float accept_scale;
	float wp_value;
	float wp_deriv;
	bool strict;	
	int max_round;
	float mg_param;
	int cg_reset_num;
	int L_save_num;
	bool debug;		
	char mod;
	float init_step;
	float step;
	float r_step;
	int dimen;
	float drv_scl;
	float max_search_angle;
	float max_step;
	char file_name[50];
	void record(float stp){			
		r_step=stp;
		if(stp==0){
			cout<<"\t[reset]";
			stp=init_step;
		}
		step=(step+stp)/2;
		
	}
	int memery(bool show=true){
		int ret=dimen*4*sizeof(float);
		int mL=0;
		coutd;
		if(mod=='4'){
			mL=sizeof(float)*dimen*L_save_num;				
			if(show)show_memery_size(mL,'g',"\n\rLBFGS算法占用");
		}
		ret+=mL;
		if(show)show_memery_size(ret,'g',"\n\r算法共计占用");
		return ret;
	}
	search_set(){
		accept_scale=0.01;
		wp_value=0.1;
		wp_deriv=0.4;
		max_round=5;
		strict=false;
		mg_param=0.5;
		cg_reset_num=50;
		L_save_num=3;
		init_step=0.1;
		max_step=2;
		reset_step();
		drv_scl=0.5;
		max_search_angle=0.05;
		debug=false;
		mod='4';
		sprintf(file_name,"%s","search_set.stl");
	}
	void show(){
		coutd<<"\t\t<搜索算法参数>";
		coutd<<"\t0.wolfe_powell:接受条件(终点与起点导数值之比)"<<accept_scale;
		coutd<<"\t1.wolfe_powell:函数值系数"<<wp_value;
		coutd<<"\t2.wolfe_powell:导数值系数"<<wp_deriv;
		coutd<<"\t3.wolfe_powell:搜索次数"<<max_round;
		coutd<<"\t4.wolfe_powell:严格条件(0:否 1：是) "<<strict;
		coutd<<"\t5.动量梯度法动量系数 "<<mg_param;
		coutd<<"\t6.共轭梯度法重置间隔"<<cg_reset_num;
		coutd<<"\t7.拟牛顿法保存矩阵行数"<<L_save_num;					
		coutd<<"\t8.初始步长"<<init_step;
		coutd<<"\t9.搜索方式:"<<mod;
		coutd<<"\t\t[1]动量梯度 [2]最速下降 [3]共轭梯度 [4]限域拟牛顿";
		coutd<<"\ta.终止梯度比例"<<drv_scl;
		coutd<<"\tb.搜索方向与梯度方向最大夹角cos值:"<<max_search_angle;
		coutd<<"\ts.最大步长:"<<max_step;
		coutd<<"\td.调试模式(0:否 1：是)"<<debug;
		coutd<<"\n\t当前步长:"<<step;			
		memery();
	}
	void set(){
		show();
		coutd<<"\tc.确定 l.查看";
		string sel;
		do{
			coutd<<"设置参数>";
			cin>>sel;
			switch(sel[0]){
			case '0':
				cin>>accept_scale;
				break;			
			case '1':
				cin>>wp_value;
				break;
			case '2':
				cin>>wp_deriv;
				break;
			case '3':
				cin>>max_round;
				break;
			case '4':
				cin>>strict;
				break;
			case '5':
				cin>>mg_param;
				break;
			case '6':
				cin>>cg_reset_num;
				break;
			case '7':
				cin>>L_save_num;
				break;
			case '8':
				cin>>init_step;
				reset_step();
				break;
			case '9':
				cin>>mod;
				reset_step();
				break;
			case 'a':
				cin>>drv_scl;
				break;
			case 'd':
				cin>>debug;
				break;
			case 's':
				cin>>max_step;
				break;
			case 'l':
				show();
				break;
			}

		}while(sel[0]!='c');
	}
	void reset_step(){
		step=init_step;
		r_step=init_step;
	}
	void save(string path){
		path=path+file_name;
		coutd<<"正在保存"<<path;
		ofstream fin(path,ios::binary);
		fin.write((char *)this,sizeof(search_set));
		fin.close();
	}
	bool read(string path){
		path=path+file_name;
		ifstream fin(path,ios::binary);
		if(!fin)return false;
		coutd<<"正在读取"<<path;
		fin.read((char *)this,sizeof(search_set));	
		fin.close();
		return true;
	}
};
class search_tool:public array_operate{
	//调用虚函数cacal()以计算pos处的值和梯度，结果分别保留在result和deriv中

public:
	bool pause_flag;
	float current_step;
	float init_deriv;
	search_tool(string path){
		pos_init=NULL;
		deriv_tmp=NULL;
		direct=NULL;
		tmp_array=NULL;
		current_step=0;
		pause_flag=false;
		root=path;
	}
	~search_tool(){
		free_mem();
	}
	void search_init(int dmn,float *rlt_p,float *pos_p,float *drv_p){
		result=rlt_p;
		pos=pos_p;
		deriv=drv_p;
		dimen=dmn;			
		if(!set.read(root))	set_search();		
		set.dimen=dmn;
		free_mem();		
		cudaMalloc((void**)&pos_init,sizeof(float)*dimen); 
		cudaMemset(pos_init,0,sizeof(float)*dimen); 
		CUDA_CHECK;
		cudaMalloc((void**)&deriv_tmp,sizeof(float)*dimen);
		cudaMemset(deriv_tmp,0,sizeof(float)*dimen); 
		CUDA_CHECK;
		cudaMalloc((void**)&direct,sizeof(float)*dimen); 
		cudaMemset(direct,0,sizeof(float)*dimen);
		CUDA_CHECK;
		cudaMalloc((void**)&tmp_array,sizeof(float)*dimen); 
		cudaMemset(tmp_array,0,sizeof(float)*dimen);
		CUDA_CHECK;	
		if(set.mod=='4')Ld.malloc(dimen,set.L_save_num);
		else Ld.free_mem();
	}

	void set_search(){
		set.dimen=dimen;
		set.set();
		if(set.mod=='4')Ld.malloc(dimen,set.L_save_num);
		else Ld.free_mem();
		set.save(root);
	}
	void reset_step(){
		set.reset_step();
		current_step=set.init_step;
	}
	void save_search(){
		set.save(root);
	}
	bool search(float r){
		pause_flag=false;
		int rounds=r;
		rounds+=((rand()%1000)<float(r-rounds)*1000)?1:0;	
		switch(set.mod){
		case '1':
			if(!momentum_grad(rounds))return false;
			break;
		case '2':
			if(!fast_grad(rounds))return false;
			break;
		case '3':
			if(!conj_grad(rounds))return false;
			break;
		case '4':
			if(!LBFGS(rounds))return false;
			break;
		}
		return true;
	}
protected:
	virtual bool show_and_control(int)=0;
	virtual void cacul()=0;
private:
	float *result;
	float *pos;
	float *deriv;	
	string root;
	search_set set;
	float *pos_init;
	float *deriv_tmp;	
	float *direct;
	float *tmp_array;
	float deriv_len;
	void free_mem(){
		if(pos_init!=NULL){
			cudaFree(pos_init);
			cudaFree(deriv_tmp);
			cudaFree(direct);
			cudaFree(tmp_array);
			CUDA_CHECK;	
		}
	}
	bool pause(){
		if(kbhit()){			
			char s=getchar();
			if(s=='p'||s=='P'){
				pause_flag=true;
				return true;
			}
		}
		return false;
	}
	float interpolation(float x,float v0,float v1,float derv0,float derv1){//两点三次插值
		float s,z;
		double w;
		s=3*(v1-v0)/x;
		z=s-derv1-derv0;
		w=z*z-derv1*derv0;
		if(w<0)return -1;
		w=sqrt(w);
		s=derv1-derv0+2*w;
		if(s==0)return -1;
		x=x*(w-derv0-z)/s;
		if(_finite(x))return x;
		return -1;
	}
	float wolfe_powell(float step){			
		float v0,v1,d0,d1;
		float max_s=0,min_s=0;			
		float len=length(direct);		
		v0=*result;
		
		d0=-dot(direct,deriv)/len;
		//coutd<<len<<" "<<d0;getchar();
		//搜索方向与梯度方向夹角不能太小，否则重置
		if(d0>-set.max_search_angle*deriv_len)return 0;

		float min_value=v0,min_step=0;
		clone(pos,pos_init);	//记录初始点,搜索失败或者结果不是搜索过程中最小值时使用

		int i;
		int flag;//-1偏小;1偏大;0满足条件
		for(i=0;i<set.max_round;i++){

			add(pos,direct,pos_init,-step/len);
			cacul();
			v1=*result;
			d1=-dot(direct,deriv)/len;


			if(v1<min_value){
				min_value=v1;
				min_step=step;
			}	

			flag=0;
			if(v1<v0&&abs(d1)<abs(d0)*set.accept_scale)break;//导数小于特定值则接受该点
			if(v1>v0+set.wp_value*step*d0){
				if(d1<0)flag=-1;
				else flag=1;
			}else {
				if(set.strict&&d1>-d0*set.wp_deriv)flag=1;
				if(d1<d0*set.wp_deriv)flag=-1;
			}
			if(v1>v0)flag=1;
			if(set.debug)//调试
				coutd<<"v0:"<<v0<<" v1:"<<v1<<" d0:"<<d0<<" d1:"<<d1<<" flg"<<flag<<" stp"<<step;

			if(flag==0)break;


			if(flag==1)max_s=step;			
			if(flag==-1)min_s=step;	

			float tmp=interpolation(step,v0,v1,d0,d1);//两点三次插值，如果不能找到插值点，返回-1
			if(tmp>min_s&&(tmp<max_s||max_s==0))step=tmp;			
			else{//如果三次插值失败，使用两点两次插值
				tmp=-d0*step/(d1-d0);
				if(_finite(tmp)&&tmp>min_s&&(tmp<max_s||max_s==0))step=tmp;
				else{//如果两点两次插值失败，使用中点插值
					if(flag==1)step=(max_s+min_s)/2;
					if(flag==-1){
						if(max_s!=0)step=(max_s+min_s)/2;
						else step=min_s*2;
					}
				}	
			}


		}
		
		if(*result>min_value){//如果最终得到的点不是搜索中的最小值，则以最小值的点作为结果
			add(pos,direct,pos_init,-min_step/len);
			cacul();		
		}	
		if(min_step>set.max_step){
			add(pos,direct,pos_init,-set.max_step/len);
			cacul();
			min_step=0;
		}
		return min_step;
	}
	bool momentum_grad(int rounds){//动量梯度法
		zero(pos_init);
		cacul();
		init_deriv=length(deriv);
		float v0;
		for(int i=0;i<rounds;i++){			
			add(pos_init,pos_init,deriv,set.mg_param);
			simple_add(pos,pos_init,-set.step);
			clone(deriv,pos_init);
			if(!show_and_control(i))return true;
			if(pause())return false;
			v0=*result;
			cacul();
			if(length(deriv)<set.drv_scl*init_deriv&&*result<v0)return true;
		}
		return true;
	}	
	bool move(){//一维搜索，并确定是否终止
		deriv_len=length(deriv);
		current_step=set.step;
		set.record(wolfe_powell(set.step));		
		if(set.r_step==0)return 0;
		return 1;
	}
	bool fast_grad(int rounds){//最速下降法
		cacul();
		init_deriv=length(deriv);
		for(int i=0;i<rounds;i++){	
			if(!show_and_control(i))return true;
			clone(deriv,direct);
			if(!move())return true;
			if(deriv_len<set.drv_scl*init_deriv)return true;
			if(pause())return false;
		}
		return true;
	}
	bool conj_grad(int rounds){//共轭梯度法
		cacul();
		init_deriv=length(deriv);
		float r,glen,clen;
		int reset=0;
		int flag=0;
		clone(deriv,direct);			
		for(int i=0;i<rounds;i++){
			if(!show_and_control(i))return true;
			if(pause())return false;
			glen=length(deriv);
			clone(direct,tmp_array);
			clone(deriv,deriv_tmp);
			if(!move())flag++;			
			else flag=0;
			if(deriv_len<set.drv_scl*init_deriv)return true;
			if(flag==2)return true;
			if(reset>set.cg_reset_num-1||flag==1){
				reset=0;
				r=0;		
				if(reset>=set.cg_reset_num-1)flag=0;
				clone(deriv,direct);
			}		
			else {
				//r算式：r=(grad1-grad0)*grad1
				clen=length(deriv);
				r=(clen*clen-dot(deriv_tmp,deriv))/(glen*glen);	
				add(direct,tmp_array,deriv,r); 	
				reset++;			
			}
		}
		return true;	
	} 
	struct LG_STRUCT{
		float *s_data;	
		float *y_data;
		float *alf;
		float *ro;
		float **s;
		float **y;
		float save_num;
		int dimen;
		LG_STRUCT(){
			s_data=NULL;
			dimen=0;
			save_num=0;
		}
		~LG_STRUCT(){
			free_mem();
		}

		void malloc(int dmn,int m){			
			if(dmn==dimen&&m==save_num)return;		
			free_mem();
			save_num=m;
			dimen=dmn;
			cudaMalloc((void**)&s_data,sizeof(float)*dimen*m);
			cudaMemset(s_data,0,sizeof(float)*dimen*m);
			CUDA_CHECK;
			cudaMalloc((void**)&y_data,sizeof(float)*dimen*m); 
			cudaMemset(y_data,0,sizeof(float)*dimen*m);
			CUDA_CHECK;
			alf=new float[m];			
			ro=new float[m];
			s=new float *[m];
			memzero(alf);
			memzero(ro);
			memzero(s);
			for(int i=0;i<m;i++)s[i]=s_data+dimen*i;
			y=new float *[m];
			for(int i=0;i<m;i++)y[i]=y_data+dimen*i;
		}
		void free_mem(){
			if(s_data!=NULL){
				cudaFree(s_data);
				cudaFree(y_data);
				CUDA_CHECK;
				delete [] alf;
				delete [] ro;
				delete [] s;
				delete [] y;
			}
			s_data=NULL;
			dimen=0;
			save_num=0;
		}

	};
	LG_STRUCT Ld;
	bool LBFGS(int rounds){
		float bt;
		int i,l,ll;
		cacul();
		init_deriv=length(deriv);		
		clone(deriv,direct);	
		int k=0;
		int m=Ld.save_num;	
		int flag=0;
		for(int kk=0;kk<rounds;kk++){	
			if(!show_and_control(kk))return true;
			if(pause())return false;
			clone(pos,pos_init);
			clone(deriv,deriv_tmp);
			if(!move())flag++;
			else flag=0;
			if(deriv_len<set.drv_scl*init_deriv)return true;
			if(flag==2)return true;
			if(flag==1){
				k=0;
				clone(deriv,direct); 
				continue;
			}
			l=(k<m)?k:m;
			if(k<m)ll=k;
			else{
				ll=m-1;
				Ld.ro[ll]=Ld.ro[0];
				Ld.s[ll]=Ld.s[0];
				Ld.y[ll]=Ld.y[0];
				for(int i=0;i<ll;i++){
					Ld.ro[i]=Ld.ro[i+1];
					Ld.s[i]=Ld.s[i+1];
					Ld.y[i]=Ld.y[i+1];					
				}
			}
			add(Ld.s[ll],pos_init,pos,-1);
			add(Ld.y[ll],deriv_tmp,deriv,-1); 
			Ld.ro[ll]=dot(Ld.s[ll],Ld.y[ll]);
			if(Ld.ro[ll]==0){
				k=0;
				clone(deriv,direct); 
				continue;
			}
			clone(deriv,tmp_array);
			for(i=l-1;i>=0;i--){
				Ld.alf[i]=dot(Ld.s[i],tmp_array)/Ld.ro[i];
				add(tmp_array,Ld.y[i],tmp_array,-Ld.alf[i]); 
			}
			for(int i=0;i<l;i++){
				bt=dot(Ld.y[i],tmp_array)/Ld.ro[i];
				add(tmp_array,Ld.s[i],tmp_array,Ld.alf[i]-bt); 
			}
			clone(tmp_array,direct);			
			k++;

		}
		return true;
	}	

};

