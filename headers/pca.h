

struct jacobi{
	float *mtx;
	float *eigen_mtx;
	float *eigen_values;
	float scale;
	int dimen;
	int stop_jugde_num;
	float *record;
	float *record_v;
	int r_round;
	int round;
	int count;
	jacobi(int dmn,float *m_mtx,float *e_mtx,float *e_value,float scl,int sjn=10){
		dimen=dmn;
		mtx=new float[dimen*dimen];
		memcpy(mtx,m_mtx,sizeof(float)*dimen*dimen);
		eigen_mtx=e_mtx;
		eigen_values=e_value;
		scale=scl;
		stop_jugde_num=sjn;//最大相关系数连续不变的次数，以此判断是否停止
		record=new float[stop_jugde_num];
		record_v=new float[stop_jugde_num];
		round=1000;
		count=0;
		r_round=0;
	}

	~jacobi(){
		delete [] mtx;
		delete [] record;
		delete [] record_v;
	}
	void givens_rot(char side,float *mt,float sinv,float cosv,int c,int r){
		float a1,a2;	
		for(int i=0;i<dimen;i++){
			if(side=='l'){
				a1=mt[c*dimen+i];
				a2=mt[r*dimen+i];
				mt[c*dimen+i]=a1*cosv-a2*sinv;
				mt[r*dimen+i]=a1*sinv+a2*cosv;
			}
			if(side=='r'){
				a1=mt[i*dimen+c];
				a2=mt[i*dimen+r];
				mt[i*dimen+c]=a1*cosv+a2*sinv;
				mt[i*dimen+r]=-a1*sinv+a2*cosv;
			}
		}
	}
	void get_sin_cos(int c,int r,float &sinv,float &cosv){
		float vcc=mtx[c*dimen+c];
		float vrr=mtx[r*dimen+r];
		float vrc=mtx[r*dimen+c];
		if(vcc==vrr){
			if(vrc<0)sinv=-sqrt(2.0f)/2;
			else sinv=sqrt(2.0f)/2;
			cosv=sqrt(2.0f)/2;
		}else{
			float y=abs(vcc-vrr);
			float x=2*vrc*((vcc>vrr)?1:-1);
			float z=sqrt(x*x+y*y);
			cosv=sqrt(0.5f*(1+y/z));
			sinv=x/z/2/cosv;
		}
	}
	float find_max_value(int &column,int &row){
		float ret=0,v_t;
		int k;
		for(int i=0;i<dimen;i++){//寻找最大的非对角主元
			k=i*dimen;
			for(int j=i+1;j<dimen;j++){				
				v_t=abs(mtx[k+j]);
				if(v_t>ret){
					ret=v_t;
					column=j;
					row=i;
					
				}
			}
		}
		if(column>row){
						int t=column;
						column=row;
						row=t;
					}
		return ret;
	}
	float find_max_correlation(int &column,int &row){
		float ret=0,v_t,vrr;
		for(int i=0;i<dimen;i++){
			vrr=mtx[i*dimen+i];
			for(int j=i+1;j<dimen;j++){
				float t=sqrt(abs(mtx[j*dimen+j]*vrr));
				if(t==0)v_t=0;
				else v_t=abs(mtx[i*dimen+j])/t;
				if(v_t>ret){
					ret=v_t;
					column=j;
					row=i;
					
				}
			}
		}
		if(column>row){
						int t=column;
						column=row;
						row=t;
					}
		return ret;
	}	
	bool is_stop(){
		int t,tt;
		float maxvv=find_max_value(t,tt);
		record_v[r_round%stop_jugde_num]= maxvv;
		float maxv=find_max_correlation(t,tt);
		coutd<<r_round<<"("<<count<<") max_cor"<<maxv<<" max_val"<<maxvv;
		record[r_round%stop_jugde_num]= maxv;
		r_round++;
		if( maxv<scale)return true;		
		for(int i=0;i<stop_jugde_num;i++){
			for(int j=i+1;j<stop_jugde_num;j++){
				if(record[i]!=record[j])return false;
				if(record_v[i]!=record_v[j])return false;
			}
		}
		return true;
	}
	static int cmp(float &a,float &b){
		if(a>b)return 1;
		return 0;
	}
	void show_eigen(){
		coutd<<"本征值(按从大到小顺序排列):"<<setprecision(4);
		coutd;
		float *st=new float[dimen];
		memcpy(st,eigen_values,_msize(st));
		sort(st,st+dimen,cmp);
		float sum=0;
		for(int i=0;i<dimen;i++){
			sum+=st[i];
		}
		float sumt=0;
		for(int i=0;i<dimen;i++){
			sumt+=st[i];
			cout<<st[i]<<"\t";
			if(i%10==9){
				coutd<<"["<<(sumt*100/sum)<<"]";
				coutd;
			}
		}
		cout<<setprecision(5);
	}

	void run(bool debug=false){		
		int row,column;
		float sinv,cosv;		
		for(int i=0;i<stop_jugde_num;i++){
			record[i]=i;
			record_v[i]=i;
		}
		memset(eigen_mtx,0,sizeof(float)*dimen*dimen);
		for(int i=0;i<dimen;i++)eigen_mtx[i*(dimen+1)]=1.0f;
		is_stop();
		float maxv;
		while(1){
			count++;
			if((count%round==round-1)){
				if(is_stop())break;
			}
			maxv=find_max_value(column,row);
			if(debug)coutd<<"max"<<maxv<<" r"<<row<<" c"<<column;
			get_sin_cos(column,row,sinv,cosv);
			if(debug)cout<<" sin"<<sinv<<" cos"<<cosv;
			givens_rot('r',mtx,sinv,cosv,column,row);
			givens_rot('l',mtx,-sinv,cosv,column,row);
			givens_rot('r',eigen_mtx,sinv,cosv,column,row);
		}
		for(int i=0;i<dimen;i++){
			eigen_values[i]=mtx[i*(dimen+1)];
		}
	}
};
class PCA{

   private:
	string pre_file;
	float *avg;
	float *pca_mtx;
	float *eigen;	
	float *dev;
	float *max_value;
	float *o_cov_mtx;//原始协方差矩阵
	void mem_free(){
		if(avg!=NULL){
			cudaFree(avg);
			delete [] max_value;
			delete [] dev;
			delete [] o_cov_mtx;
		}
		safe_free(eigen);
		safe_gpu_free(pca_mtx);
		if(add_noise!=NULL)delete add_noise;
		avg=NULL;
		eigen=NULL;
		pca_mtx=NULL;
		add_noise=NULL;
	}
	struct pca_set{
		int dimen;
		int data_num;
		int group_num;
		float pca_redc_scl;//PCA维尺度
		float jacobi_scl;
		int pca_dmn;
		int jacobi_stop_num;
		bool debug;
		bool scaled;
		bool dev_scaled;
		bool pca;
		bool pca_scaled;
		bool pre_operate;
		pca_set(){
			data_num=6000;
			group_num=10;
			jacobi_scl=0.0001;
			pca_redc_scl=0.999;
			jacobi_stop_num=20;
			debug=false;
			scaled=false;
			dev_scaled=false;
			pca=false;
			pca_scaled=false;
			pre_operate=true;
			pca_dmn=0;
		}
		void show_pca(){		
			coutd<<"\t<数据预处理参数>";
			coutd<<"\t输入维度:"<<dimen;
			coutd<<"\t降维维度:"<<pca_dmn;
			coutd<<"\t预处理数据[1.是 0.否]:"<<pre_operate;
			coutd<<"\t数据分组数量:"<<group_num;
			coutd<<"\t每组采样数量:"<<data_num;
			coutd<<"\t归一化策略:"<<(dev_scaled?"方差":(scaled?"量纲":"不归一化"));
			coutd<<"\tPCA处理[1.是 0.否]"<<pca;
			coutd<<"\t雅可比法终止尺度"<<jacobi_scl;
			coutd<<"\t雅可比法终止判断轮次"<<jacobi_stop_num;
			coutd<<"\tPCA降维尺度"<<pca_redc_scl;
			coutd<<"\tPCA白化处理[1.是 0.否]"<<pca_scaled;
			coutd;
		}	
	};
	
	void save_pre_data(bool un_end=false){
		ofstream fin(pre_file,ios::binary);
		coutd<<"正在保存"<<pre_file;
		fin.write((char *)&set,sizeof(pca_set));
		if(!set.pre_operate){
			fin.close();
			return;
		}
		float *t_avg=new float[set.dimen];
		cudaMemcpy(t_avg,avg,_msize(t_avg),cudaMemcpyDeviceToHost);
		fin.write((char *)t_avg,_msize(t_avg));
		fin.write((char *)dev,_msize(dev));
		fin.write((char *)max_value,_msize(max_value));
		fin.write((char *)o_cov_mtx,_msize(o_cov_mtx));	
		delete [] t_avg;
		if(un_end){
			fin.close();
			return;
		}
		float *tmp=new float[set.dimen*set.pca_dmn];
		cudaMemcpy(tmp,pca_mtx,sizeof(float)*set.dimen*set.pca_dmn,cudaMemcpyDeviceToHost);
		fin.write((char *)tmp,sizeof(float)*set.dimen*set.pca_dmn);			
		delete [] tmp;
		if(set.pca)fin.write((char *)eigen,sizeof(float)*set.pca_dmn);			
		fin.close();
		
	}
	void read_pre_data(bool un_end=false){
		mem_free();
		ifstream fin(pre_file,ios::binary);
		coutd<<"正在读取"<<pre_file;
		fin.read((char *)&set,sizeof(pca_set));
		if(!set.pre_operate){
			fin.close();
			return;
		}
		float *t_avg=new float[set.dimen];		
		cudaMalloc((void **)&avg,_msize(t_avg));
		dev=new  float[set.dimen];
		max_value=new  float[set.dimen];
		o_cov_mtx=new  float[set.dimen*set.dimen];
	
		fin.read((char *)t_avg,_msize(t_avg));
		cudaMemcpy(avg,t_avg,_msize(t_avg),cudaMemcpyHostToDevice);
		CUDA_CHECK;	
		fin.read((char *)dev,_msize(dev));
		fin.read((char *)max_value,_msize(max_value));
		fin.read((char *)o_cov_mtx,_msize(o_cov_mtx));		
		delete [] t_avg;
		if(un_end){
			fin.close();
			return;
		}
		set.show_pca();
		add_noise=new noise(set.dimen,dev);
		cudaMalloc((void **)&pca_mtx,sizeof(float)*set.pca_dmn*set.dimen);
		float *tmp=new float[set.pca_dmn*set.dimen];
		fin.read((char *)tmp,sizeof(float)*set.pca_dmn*set.dimen);
		cudaMemcpy(pca_mtx,tmp,sizeof(float)*set.pca_dmn*set.dimen,cudaMemcpyHostToDevice);
		CUDA_CHECK;	
		delete []tmp;
		if(set.pca){
			eigen=new float[set.pca_dmn];
			fin.read((char *)eigen,sizeof(float)*set.pca_dmn);
		}
		fin.close();
	}
	void show_mtx(float *m,int dimen){
		float maxv=0,minv=100000000;
		for(int i=0;i<dimen;i++){
			coutd<<"[第"<<i<<"行]";
			for(int j=0;j<dimen;j++){
				float v=m[i*dimen+j];
				cout<<" "<<v;
				if(abs(v)>maxv)maxv=abs(v);
				if(abs(v)<minv)minv=abs(v);
			}
			if(i%10==9)getchar();
		}
		coutd<<"最大绝对值"<<maxv<<" "<<"最小绝对值"<<minv;
		getchar();
	}
	void show_correlation(){
		//查看相关系数
		float maxv,minv,vrr,v_t;
		int imax1,imax2,imin1,imin2;
		coutd<<"平均值";
		maxv=-1000,minv=1000;
		for(int i=0;i<set.dimen;i++){
			v_t=gpu_read_unit_value(avg+i);
			cout<<"["<<i<<"]"<<v_t<<"\t";
			if(v_t>maxv){
					maxv=v_t;	
					imax1=i;
				}
				if(v_t<minv){
					minv=v_t;	
					imin1=i;
				}
			
		}
		coutd<<"最大平均值:"<<maxv<<"["<<imax1<<"]"<<"\t最小平均值"<<minv<<"["<<imin1<<"]";
		coutd<<"标准差";
		maxv=-1000,minv=1000;
		for(int i=0;i<set.dimen;i++){
			v_t=dev[i];
			cout<<"["<<i<<"]"<<v_t<<"\t";
			if(v_t>maxv){
					maxv=v_t;	
					imax1=i;
				}
				if(v_t<minv){
					minv=v_t;	
					imin1=i;
				}
			
		}
		coutd<<"最大标准差:"<<maxv<<"["<<imax1<<"]"<<"\t最小标准差"<<minv<<"["<<imin1<<"]";
		getchar();
		maxv=-1,minv=1;
		coutd;
		float scl;
		coutd<<"显示的相关系数的绝对值下限:";
		cin>>scl;
		for(int i=0;i<set.dimen;i++){
			maxv=-1,minv=1;
			vrr=o_cov_mtx[i*set.dimen+i];
			bool bl=false;
			for(int j=i+1;j<set.dimen;j++){
				v_t=o_cov_mtx[i*set.dimen+j]/sqrt(abs(o_cov_mtx[j*set.dimen+j]*vrr));
				if(abs(v_t)>scl){
					cout<<"["<<i<<","<<j<<"]"<<v_t<<"\t";
					bl=true;
				}
				if(v_t>maxv){
					maxv=v_t;	
					imax1=i;
					imax2=j;
				}
				if(v_t<minv){
					minv=v_t;	
					imin1=i;
					imin2=j;
				}
			}
			if(bl)getchar();
		}
	}
	void sampling_normalization(){
		mem_free();
		coutd<<"输入数据采样次数和每组采样数量(空格分隔）:";
		cin>>set.group_num>>set.data_num;
		coutd<<"选择归一化策略:[0.不归一化 1.归一化量纲 2.归一化方差]";
		string s;
		cin>>s;
		set.scaled=0;set.dev_scaled=1;
		if(s[0]=='1'){
			set.scaled=1;
			set.dev_scaled=0;
		}
		if(s[0]=='0'){
			set.scaled=0;
			set.dev_scaled=0;
		}

		cudaMalloc((void**)&avg,sizeof(float)*set.dimen);//均值
		dev=new float[set.dimen];//方差
		max_value=new float[set.dimen];//最大绝对值
		o_cov_mtx=new float[set.dimen*set.dimen];
		
		double *maxv=new double[set.dimen];//最大绝对值
		double *d_dev=new double[set.dimen];//最大绝对值
		double *tmp_cov_mtx;//协方差矩阵
		float *data;
		float *t_avg;
		float *t_dev=new float[set.dimen];	
		double *tmp_data;
		cudaMalloc((void**)&tmp_cov_mtx,sizeof(double)*set.dimen*set.dimen);		
		memzero(maxv);
		memzero(d_dev);		
		cudaMemset(avg,0,sizeof(float)*set.dimen); 
		cudaMemset(tmp_cov_mtx, 0,sizeof(double)*set.dimen*set.dimen);
		cudaMalloc((void**)&data,sizeof(float)*set.dimen*set.data_num);			
		cudaMalloc((void**)&t_avg,sizeof(float)*set.dimen);		
		cudaMalloc((void**)&tmp_data,sizeof(double)*set.dimen*set.data_num);	
		array_group_sum a_sum(set.dimen,set.data_num);
		for(int t=0;t<set.group_num;t++){	
			coutd<<"第"<<t<<"次采样...";
			get_pre_data(data,set.data_num);			
			CUDA_CHECK;	
			//计算均值
			cudaMemset(t_avg ,0,sizeof(float)*set.dimen);  
			CUDA_CHECK;				
			a_sum.sum(t_avg,data);			
			float alpha=1.0f/set.data_num;
			CUBT(cublasSscal(cublasHandle,set.dimen, &alpha,t_avg, 1));	
			alpha=1.0f;
			CUBT(cublasSaxpy(cublasHandle,set.dimen,&alpha,t_avg, 1, avg, 1));
			//计算方差
			array_add_to_matrix(data,t_avg,-1,set.dimen,set.data_num);					
			float tt;int idx;
			for(int j=0;j<set.dimen;j++){
				if(set.scaled){
					CUBT(cublasIsamax(cublasHandle,set.data_num, data+j, set.dimen, &idx));
					maxv[j]+=abs(gpu_read_unit_value(data+(idx-1)*set.dimen+j));//获取每组数据最大值的均值,用于量纲归一化
				}else maxv[j]+=1.0f;
				CUBT(cublasSnrm2(cublasHandle, set.data_num, data+j, set.dimen, t_dev+j));						
			}
			float snm=1.0f/sqrt(float(set.data_num));	
			for(int i=0;i<set.dimen;i++){
				t_dev[i]*=snm;		
				d_dev[i]+=t_dev[i];				
				if(set.dev_scaled){
					//对原数据归一化方差
					alpha=1.0f/((t_dev[i]>0)?t_dev[i]:1);
					CUBT(cublasSscal(cublasHandle,set.data_num, &alpha,data+i,set.dimen));	
				}
			}
			//计算协方差矩阵，计算出的矩阵还未经过量纲归一化，如果选择了方差归一化，则无需进行量纲归一化
			array_type_trans(data,tmp_data,set.dimen*set.data_num);			
			double alphad=1.0f;	
			for(int j=0;j<set.data_num;j++){
					CUBT(cublasDsyr(cublasHandle, CUBLAS_FILL_MODE_UPPER, set.dimen,&alphad,tmp_data+j*set.dimen, 1, tmp_cov_mtx,set.dimen));				
			}
			
		}

		//计算平均均值、平均方差
		float alpha=1.0f/set.group_num;
		CUBT(cublasSscal(cublasHandle,set.dimen, &alpha,avg,1));
		for(int i=0;i<set.dimen;i++){
			dev[i]=d_dev[i]*alpha;
			max_value[i]=maxv[i]*alpha;
			if(max_value[i]==0)max_value[i]=1;			
		}
	
		//计算平均协方差矩阵
		float *g_cov_mtx;		
		cudaMalloc((void**)&g_cov_mtx,sizeof(float)*set.dimen*set.dimen);
		array_type_trans(tmp_cov_mtx,g_cov_mtx,set.dimen*set.dimen);	
		alpha=1.0f/set.data_num/set.group_num;
		CUBT(cublasSscal(cublasHandle,set.dimen*set.dimen, &alpha,g_cov_mtx,1));		
		cudaMemcpy(o_cov_mtx,g_cov_mtx,sizeof(float)*set.dimen*set.dimen,cudaMemcpyDeviceToHost);
		CUDA_CHECK;	
		cudaFree(g_cov_mtx);			
		CUDA_CHECK;
		mtr_semi2all(o_cov_mtx,set.dimen,false);
		//show_correlation();
		delete [] maxv;//最大绝对值
		delete [] d_dev;//最大绝对值
		delete [] t_dev;
		cudaFree(tmp_cov_mtx);//协方差矩阵
		cudaFree(data);
		cudaFree(t_avg);		
		cudaFree(tmp_data);
		save_pre_data(true);
	}
	void make_pre_mtx(float *pre_data_mtx){
		cudaMemset(pre_data_mtx,0,sizeof(float)*set.dimen*set.dimen);		
		for(int i=0;i<set.dimen;i++){
			float v;
			if(set.dev_scaled){
				v=(dev[i]>0.0000001f)?(1.0f/dev[i]):0;
			}else{
				v=1.0f/max_value[i];
			}
			gpu_write_unit_value(pre_data_mtx+i*set.dimen+i,v);
		}
		CUDA_CHECK;	
	}
	void unpca(){
		set.pca_dmn=set.dimen;
		cudaMalloc((void**)&pca_mtx,sizeof(float)*set.pca_dmn*set.dimen);
		make_pre_mtx(pca_mtx);
		save_pre_data();
	}
	void pca(){
		read_pre_data(true);		
		set.pca=1;
		//coutd<<"输入雅克比法终止的尺度和判断终止的轮次:";
	//	coutd<<"(空格分隔，当前值分别为:"<<set.jacobi_scl<<"  "<<set.jacobi_stop_num<<")";
		set.jacobi_scl=0.01;
		set.jacobi_stop_num=5;
		cin>>set.jacobi_scl>>set.jacobi_stop_num;
		
		float *cov_mtx=new float[set.dimen*set.dimen];	
		for(int i=0;i<set.dimen;i++){
			float scl_i=(set.dev_scaled)?1.0f:max_value[i];
			cov_mtx[i*set.dimen+i]=o_cov_mtx[i*set.dimen+i]/scl_i/scl_i;
			for(int j=i+1;j<set.dimen;j++){
				float scl_j=(set.dev_scaled)?1.0f:max_value[j];
				cov_mtx[j*set.dimen+i]=o_cov_mtx[j*set.dimen+i]/scl_i/scl_j;
				cov_mtx[i*set.dimen+j]=o_cov_mtx[i*set.dimen+j]/scl_i/scl_j;
			}
		}			
		
		
		//计算本征值
		float *eigen_vector_mtx=new float[set.dimen*set.dimen];
		float *eigen_values=new float[set.dimen];
		//show_mtx(cov_mtx,set.dimen);
		coutd<<"正在求解本征值";	
		jacobi j(set.dimen,cov_mtx,eigen_vector_mtx,eigen_values,set.jacobi_scl,set.jacobi_stop_num);
		j.run(set.debug);
		j.show_eigen();
		coutd<<"输入保留本征值的比例:";
		cin>>set.pca_redc_scl;
		float scl=get_scale(eigen_values,set.dimen,set.pca_redc_scl);	
		int *list=new int[set.dimen];
		set.pca_dmn=0;
		for(int i=0;i<set.dimen;i++){//记录要保留的维度
				if(eigen_values[i]>=scl){				
					list[set.pca_dmn]=i;
					set.pca_dmn++;			
				}
		}
		coutd<<"保留维数:"<<set.pca_dmn;	
		float *eigen_vectors=new float[set.pca_dmn*set.dimen];
		eigen=new float[set.pca_dmn];
		for(int i=0;i<set.pca_dmn;i++){
				float scale=(set.pca_scaled)?(1.0f/sqrt(eigen_values[list[i]])):1.0f;
				eigen[i]=(set.pca_scaled)?1.0f:sqrt(eigen_values[list[i]]);
				for(int j=0;j<set.dimen;j++){
					eigen_vectors[i*set.dimen+j]=eigen_vector_mtx[j*set.dimen+list[i]]*scale;
				}			
		}
		float *pre_data_mtx;
		cudaMalloc((void**)&pre_data_mtx,sizeof(float)*set.dimen*set.dimen);
		make_pre_mtx(pre_data_mtx);
		cudaMalloc((void**)&pca_mtx,sizeof(float)*set.pca_dmn*set.dimen);	
		float *tmp_eigen_vectors;
		cudaMalloc((void**)&tmp_eigen_vectors,sizeof(float)*set.pca_dmn*set.dimen);
		cudaMemcpy(tmp_eigen_vectors,eigen_vectors,sizeof(float)*set.pca_dmn*set.dimen,cudaMemcpyHostToDevice);
		CUDA_CHECK;
		float alpha=1.0f;
		float beta=0;
		CUBT(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,set.dimen, set.pca_dmn,set.dimen, &alpha,pre_data_mtx,set.dimen,tmp_eigen_vectors, set.dimen, &beta,pca_mtx, set.dimen));			
		delete [] eigen_values;
		delete [] eigen_vectors;
		delete [] cov_mtx;
		delete [] list;	
		delete [] eigen_vector_mtx;
		cudaFree(tmp_eigen_vectors);
		cudaFree(pre_data_mtx);
		CUDA_CHECK;
		save_pre_data();

	};

	void mtr_semi2all(float *mtx,int dimen,bool cuda=true){
		for(int i=0;i<dimen-1;i++){
			for(int j=i+1;j<dimen;j++){
				if(cuda)cudaMemcpy(mtx+i*dimen+j,mtx+j*dimen+i,sizeof(float),cudaMemcpyDeviceToDevice);
				else mtx[i*dimen+j]= mtx[j*dimen+i];
			}
		}
		CUDA_CHECK;
	}	
	static int cmp(float &a,float &b){
		if(a<b)return 1;
		return 0;
	}
	float get_scale(float *arr,int dimen,float scl){
		float *st=new float[dimen];
		double total_val=0;
		for(int i=0;i<dimen;i++){
			st[i]=arr[i];
			total_val+=st[i];
		}
		sort(st,st+dimen,cmp);
		double total_dev=0;
		float ret=0;
		for(int i=0;i<dimen;i++){			
			if(total_dev>=(1-scl)*total_val)break;
			ret=st[i];
			total_dev+=st[i];
		}
		delete [] st;
		return ret;
	}


	float trans_alpha;
	float trans_beta;	
	struct noise{
		curandGenerator_t gen;
		int dimen;
		int num;
		int r_num;
		float *rnd;
		float *dev;
		float alpha;
		int real_r_num;
		noise(int dmn,float *dv){
			CHECK_CURAND( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
			CHECK_CURAND( curandSetPseudoRandomGeneratorSeed(gen,time(NULL)));
			dimen=dmn;
			dev=dv;
			num=0;
			alpha=1.0f;
			rnd=NULL;
		}
		~noise(){
			safe_gpu_free(rnd);
			CHECK_CURAND( curandDestroyGenerator(gen) );
		}
		void set_num(int data_num){
			num=data_num;
			safe_gpu_free(rnd);
			real_r_num=dimen*num;	
			r_num=real_r_num;
			r_num=(r_num%2==1)?(r_num+1):r_num;//curand生成的随机数必须是偶数个
			cudaMalloc( (void **) &rnd,r_num* sizeof(float) ) ;
			
		}
		void add(float *data,int data_num,float noise_scale){
			 if(data_num!=num)set_num(data_num);
			 CHECK_CURAND( curandGenerateNormal(gen, rnd,r_num,0,noise_scale) );
			 for(int i=0;i<dimen;i++){
				CUBT(cublasSscal(cublasHandle,data_num, dev+i,rnd+i,dimen));
			}
			CUBT(cublasSaxpy(cublasHandle,real_r_num,&alpha, rnd, 1,data, 1));
			
		}
	};
	noise *add_noise;
		public:

	pca_set set;
	PCA(string path=""){
		pre_file=path+"pre_PCA.stl";
		avg=NULL;
		eigen=NULL;
		add_noise=NULL;
		pca_mtx=NULL;
		trans_alpha=1.0f;
		trans_beta=0;
		g_gpu_init();
		
	}
	~PCA(){
		mem_free();
	}
	bool pre_read(){
		ifstream fin(pre_file,ios::binary);
		if(!fin)return false;
		fin.close();
		read_pre_data();
	}
	void pca_main(int dmn,bool reset=false){	
		set.dimen=dmn;
		//coutd<<"是否预处理数据?[1.是 0.否]";[EDIT_SYMBOL]
		//cin>>set.pre_operate;		
		set.pre_operate=0;
		if(!set.pre_operate){
			set.pca_dmn=set.dimen;
			save_pre_data();
			return ;
		}
		if(!reset)sampling_normalization();
		do{	
			string s;
			coutd<<"选择:";
			
			coutd<<"\t1.重新采样:";
			coutd<<"\t2.PCA计算:";
			coutd<<"\t3.查看采样结果";
			coutd<<"\tc.完成:";
			cin>>s;
			
			if(s[0]=='1'){
				set.pca=0;
				sampling_normalization();
			}
			if(s[0]=='2'){				
				pca();
			}
			if(s[0]=='3'){
				show_correlation();
			}
			if(s[0]=='c'){
				if(set.pca==0){
					coutd<<"是否不进行PCA处理?(Y/N)";
					cin>>s;
					if(s[0]=='y'||s[0]=='Y'){
						unpca();
						break;
					}
				}else{
					break;
				}
			}
		}while(1);
	}
	void trans_data(float *from,float *to,int num,float noise_scl=0){
		if(!set.pre_operate){
			cudaMemcpy(to,from,sizeof(float)*num*set.dimen,cudaMemcpyDeviceToDevice);
			return;
		}
		array_add_to_matrix(from,avg,-1,set.dimen,num);
		if(noise_scl>0){
			add_noise->add(from,num,noise_scl);
		}
	
		CUBT(cublasSgemm(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_N,set.pca_dmn,num,set.dimen, &trans_alpha,
			pca_mtx,set.dimen,from ,set.dimen,&trans_beta,to,set.pca_dmn));
			
	}
	virtual void get_pre_data(float *,int)=0;
};


class virtual_data_set{
public:
	int input_dimen;
	int output_dimen;
	string action_illu;
	int *bagging_idx;
	int *compare_idx;
	bool bag_mod;
	int scl_from;
	int scl_to;
	int scl;
	int compare_scl_from;
	int compare_scl_to;
	int compare_scl;
	int sample_num;
	struct cmp_data{
		int num;
		int input_num;
		int output_num;
		float *trans_input;
		float *input;
		float *target;
		float *output;
		float *cpu_input;
		float *cpu_target;
		float *cpu_output;
		float result;
		float init_result;
		void memfree(){
			if(input!=NULL){
				cudaFree(input);
				cudaFree(target);
				cudaFree(output);
				CUDA_CHECK;
				delete [] cpu_input;
				delete [] cpu_target;
				delete [] cpu_output;
				
			}
		};
		void init(int n,int input_dimen,int output_dimen){
			num=n;
			input_num=input_dimen;
			output_num=output_dimen;
			memfree();
			cudaMalloc((void**)&input,sizeof(float)*n*input_dimen);
			CUDA_CHECK;
			cudaMalloc((void**)&target,sizeof(float)*n*output_dimen);
			CUDA_CHECK;
			cudaMalloc((void**)&output,sizeof(float)*n*output_dimen);
			CUDA_CHECK;
			cpu_input=new float[input_dimen*num];
			cpu_target=new float[output_dimen*num];
			cpu_output=new float[output_dimen*num];
			
		}
		void init_data(){
			cudaMemcpy(input,cpu_input,sizeof(float)*input_num*num,cudaMemcpyHostToDevice);
			CUDA_CHECK;		
			cudaMemcpy(target,cpu_target,sizeof(float)*output_num*num,cudaMemcpyHostToDevice);
			CUDA_CHECK;	
			float avg=0;
			for(int i=0;i<num;i++){
				avg+=cpu_target[i];
			}
			avg/=num;
			init_result=0;
			for(int i=0;i<num;i++){
				init_result+=(cpu_target[i]-avg)*(cpu_target[i]-avg);
			}
			init_result/=num;
		}
		void gpu2cpu(){
			cudaMemcpy(cpu_output,output,sizeof(float)*output_num*num,cudaMemcpyDeviceToHost);
			CUDA_CHECK;
		}
		string report(){
			if(num==0)return "[]";
			int n=num;
			if(n>10000)n=10000;
			gpu2cpu();	
			string bl="",bll;
			int idx;
			stringstream f;
			for(int i=0;i<n;i++){
				f<<bl<<"[";
				bll="";
				for(int j=0;j<output_num;j++){
					idx=i*output_num+j;
					f<<bll<<"["<<cpu_target[idx]<<","<<cpu_output[idx]<<"]";
					bll=",";
				}
				bl=",";
				f<<"]";
			}
			return f.str();
		}
		cmp_data(){
			input=NULL;
			target=NULL;
			output=NULL;
			cpu_input=NULL;
			cpu_target=NULL;
			cpu_output=NULL;
			num=0;
			result=0;
			trans_input=NULL;
		}
		~cmp_data(){
			 memfree();
			 if(trans_input!=NULL)cudaFree(trans_input);
		}
	};
	virtual_data_set(){
		bagging_idx=NULL;
		compare_idx=NULL;
		bag_mod=false;
		scl_from=-1;
		sample_num=0;
	}
	~virtual_data_set(){
		if(bagging_idx!=NULL){
			delete [] bagging_idx;
			delete [] compare_idx;
		}
	}
	cmp_data compare_data;
	cmp_data train_cmp_data;
	void get_compare_data(int num,char mod='c'){
		cmp_data &s=(mod=='c')?compare_data:train_cmp_data;
		s.init(num,input_dimen,output_dimen);
		get_data(mod,s.cpu_input,s.cpu_target,num);	
		s.init_data();
		
	}
	void sample_scale_set(int sfm,int sto,int csfm,int scto){
		scl_from=sfm;
		scl_to=sto;
		compare_scl_from=csfm;
		compare_scl_to=scto;
		scl=sto-sfm;
		compare_scl=compare_scl_to-compare_scl_from;
	}
	void set_bagging_data_num(int num){
		if(sample_num==num)return;
		sample_num=num;
		if(bagging_idx!=NULL){
			delete [] bagging_idx;
			delete [] compare_idx;
		}
		bagging_idx=new int[num];
		compare_idx=new int [num];
	}
	int rand_idx(char mod='t'){
		if(!bag_mod){
			if(mod!='c')return Mrand%scl+scl_from;
			else return Mrand%(compare_scl)+compare_scl_from;
		}else{
			if(mod!='c')return bagging_idx[Mrand%sample_num];
			else return compare_idx[Mrand%sample_num];
		}
	}

	void bagging_set(int num=0){
		if(num==0)num=scl;
		bag_mod=true;
		set_bagging_data_num(num);
		int *rec=new int[scl];
		memset(rec,0,sizeof(int)*scl);
		for(int i=0;i<num;i++){
			bagging_idx[i]=Mrand%scl+scl_from;
			rec[bagging_idx[i]-scl_from]=1;
		}
		for(int i=0;i<num;i++){
			int idx;
			do{
				idx=Mrand%scl;
			}while(rec[idx]);
			compare_idx[i]=idx+scl_from;
		}
		delete [] rec;
	}
	virtual void gpu_distortion(float *input,int num)=0;
	virtual void show_compare(char mod='c')=0;
	virtual void self_action(void *mlp)=0;
	virtual void get_data(char mod,float *input,float *target,int data_num)=0;
};
