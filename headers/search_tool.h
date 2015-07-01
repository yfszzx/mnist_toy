
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
	void simple_add(float *arr1,float *arr2,float x){//arr1=arr1+arr2*x ,�˺�����add()�и�����ٶ�
		CUBT(cublasSaxpy (cublasHandle,dimen,&x,arr2,1, arr1,1));
	}
	float dot(float *arr1,float *arr2){//ʸ�����
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
class search_tool:public array_operate{
	//�����麯��cacal()�Լ���pos����ֵ���ݶȣ�����ֱ�����result��deriv��

public:
	float *result;
	float *pos;
	float *deriv;	
	search_tool(){
		pos_init=NULL;
		deriv_tmp=NULL;
		direct=NULL;
		tmp_array=NULL;
	}
	~search_tool(){
		 free_mem();
	}
	void search_init(int dmn=0,float *rlt_p=NULL,float *pos_p=NULL,float *drv_p=NULL){		
		if(dmn==0)dmn=dimen;
		if(rlt_p!=NULL)result=rlt_p;
		if(pos_p!=NULL)pos=pos_p;
		if(drv_p!=NULL)deriv=drv_p;	
		dimen=dmn;	
		set.dimen=dmn;
		free_mem();		
		cudaMalloc((void**)&pos_init,sizeof(float)*dimen); 
		CUDA_CHECK;
		cudaMalloc((void**)&deriv_tmp,sizeof(float)*dimen);
		CUDA_CHECK;
		cudaMalloc((void**)&direct,sizeof(float)*dimen); 
		CUDA_CHECK;
		cudaMalloc((void**)&tmp_array,sizeof(float)*dimen); 
		CUDA_CHECK;	
	//	set_init();
	}
	bool set_init(){
		//����ʼ�����߸ı�ѵ������֮�󣬵���search()֮ǰ����Ҫִ�д˺���һ��
		if(!set.is_setted())return false;
		set.step=set.init_step;
		if(set.mod=='4'){
			Ld.malloc(dimen,set.L_save_num);
			set.LBGFS_reset=false;
		}
		return true;
	}
	void set_search(){
		set.set();
		set_init();
	}
	virtual bool show_and_control(int)=0;
	virtual bool pause_action()=0;
	virtual void cacul()=0;
	bool search(float r){
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
				if(set.LBGFS_reset){
					Ld.malloc(dimen,set.L_save_num);
					set.LBGFS_reset=false;
					set.memery();
				}
				if(!LBFGS(rounds))return false;
				break;
			}
		return true;
	}
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
		bool setted;
		int dimen;
		bool LBGFS_reset;
		float drv_scl;
		float max_search_angle;
		void record(float stp){			
			r_step=stp;
			if(stp==0){
				coutd<<"[reset]";
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
				if(show)show_memery_size(mL,'g',"\n\rLBFGS�㷨ռ��");
			}
			ret+=mL;
			if(show)show_memery_size(ret,'g',"\n\r�㷨����ռ��");
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
			reset_step();
			drv_scl=0.5;
			max_search_angle=0.05;
			debug=false;
			mod='4';
			setted=false;	
			LBGFS_reset=false;
		}
		bool is_setted(){
			if(setted==false)coutd<<"��δ����ѵ������";
			return setted;
		}
		void show(){
			coutd<<"\t\t<�����㷨����>";
			coutd<<"\t0.wolfe_powell:��������(�յ�����㵼��ֵ֮��)"<<accept_scale;
			coutd<<"\t1.wolfe_powell:����ֵϵ��"<<wp_value;
			coutd<<"\t2.wolfe_powell:����ֵϵ��"<<wp_deriv;
			coutd<<"\t3.wolfe_powell:��������"<<max_round;
			coutd<<"\t4.wolfe_powell:�ϸ�����(0:�� 1����) "<<strict;
			coutd<<"\t5.�����ݶȷ�����ϵ�� "<<mg_param;
			coutd<<"\t6.�����ݶȷ����ü��"<<cg_reset_num;
			coutd<<"\t7.��ţ�ٷ������������"<<L_save_num;					
			coutd<<"\t8.��ʼ����"<<init_step;
			coutd<<"\t9.������ʽ:"<<mod;
			coutd<<"\t\t[1]�����ݶ� [2]�����½� [3]�����ݶ� [4]������ţ��";
			coutd<<"\ta.��ֹ�ݶȱ���"<<drv_scl;
			coutd<<"\tb.�����������ݶȷ������н�cosֵ:"<<max_search_angle;
			coutd<<"\td.����ģʽ(0:�� 1����)"<<debug;
			coutd<<"\n\t��ǰ����:"<<step;
			
			memery();
		}
		void set(){
			show();
			coutd<<"\tc.ȷ�� l.�鿴";
			string sel;
			do{
				coutd<<"���ò���>";
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
					LBGFS_reset=true;
					break;
				case '8':
					cin>>init_step;
					reset_step();
					break;
				case '9':
					cin>>mod;
					if(mod=='4')LBGFS_reset=true;
					step=init_step;
					break;
				case 'a':
					cin>>drv_scl;
					break;
				case 'd':
					cin>>debug;
					break;
				case 'l':
					show();
					break;
				}
		
			}while(sel[0]!='c');
			setted=true;
		}
		void reset_step(){
			step=init_step;
			r_step=init_step;
		}
	};
	search_set set;
private:
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
			if(s=='p'||s=='P')return pause_action();
		}
		return false;
	}
	float interpolation(float x,float v0,float v1,float derv0,float derv1){//�������β�ֵ
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
		//�����������ݶȷ���нǲ���̫С����������
		if(d0>-set.max_search_angle*deriv_len)return 0;

		float min_value=v0,min_step=0;
		clone(pos,pos_init);	//��¼��ʼ��,����ʧ�ܻ��߽������������������Сֵʱʹ��

		int i;
		int flag;//-1ƫС;1ƫ��;0��������
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
			if(v1<v0&&abs(d1)<abs(d0)*set.accept_scale)break;//����С���ض�ֵ����ܸõ�
			if(v1>v0+set.wp_value*step*d0){
				if(d1<0)flag=-1;
				else flag=1;
			}else {
				if(set.strict&&d1>-d0*set.wp_deriv)flag=1;
				if(d1<d0*set.wp_deriv)flag=-1;
			}
			if(v1>v0)flag=1;
			if(set.debug)//����
				coutd<<"v0:"<<v0<<" v1:"<<v1<<" d0:"<<d0<<" d1:"<<d1<<" flg"<<flag<<" stp"<<step;
				
			if(flag==0)break;

				
			if(flag==1)max_s=step;			
			if(flag==-1)min_s=step;	

			float tmp=interpolation(step,v0,v1,d0,d1);//�������β�ֵ����������ҵ���ֵ�㣬����-1
			if(tmp>min_s&&(tmp<max_s||max_s==0))step=tmp;			
			else{//������β�ֵʧ�ܣ�ʹ���������β�ֵ
				tmp=-d0*step/(d1-d0);
				if(_finite(tmp)&&tmp>min_s&&(tmp<max_s||max_s==0))step=tmp;
				else{//����������β�ֵʧ�ܣ�ʹ���е��ֵ
					if(flag==1)step=(max_s+min_s)/2;
					if(flag==-1){
						if(max_s!=0)step=(max_s+min_s)/2;
						else step=min_s*2;
					}
				}	
			}
				
			
		}
		
		if(*result>min_value){//������յõ��ĵ㲻�������е���Сֵ��������Сֵ�ĵ���Ϊ���
				add(pos,direct,pos_init,-min_step/len);
				cacul();		
		}		
		return min_step;
	}
	bool momentum_grad(int rounds){//�����ݶȷ�
		zero(pos_init);
		cacul();
		float init_deriv=length(deriv);
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
	bool move(){//һά��������ȷ���Ƿ���ֹ
		deriv_len=length(deriv);
		set.record(wolfe_powell(set.step));
		if(set.r_step==0)return 0;
		return 1;
	}
	bool fast_grad(int rounds){//�����½���
		cacul();
		float init_deriv=length(deriv);
		for(int i=0;i<rounds;i++){	
			if(!show_and_control(i))return true;
			clone(deriv,direct);
			if(!move())return true;
			if(deriv_len<set.drv_scl*init_deriv)return true;
			if(pause())return false;
		}
		return true;
	}
	bool conj_grad(int rounds){//�����ݶȷ�
		cacul();
		float init_deriv=length(deriv);
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
				//r��ʽ��r=(grad1-grad0)*grad1
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
		LG_STRUCT(){
			s_data=NULL;
		}
		~LG_STRUCT(){
			free_mem();
		}
		
		void malloc(int dimen,int m){
			free_mem();
			save_num=m;
			cudaMalloc((void**)&s_data,sizeof(float)*dimen*m); 
			CUDA_CHECK;
			cudaMalloc((void**)&y_data,sizeof(float)*dimen*m); 
			CUDA_CHECK;
			alf=new float[m];
			ro=new float[m];
			s=new float *[m];
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
		}

	};
	LG_STRUCT Ld;
	bool LBFGS(int rounds){
		float bt;
		int i,l,ll;
		cacul();
		float init_deriv=length(deriv);		
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
						z/=sqrt(data_num_increase_rate);
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