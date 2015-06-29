	char train_mod;	
	struct train_set_struct{
		char loss_mod;
		float noise_scale;
		char decay_mod;
		float decay;
		int count;
		int data_num;
		float T;
		int save_freq;
		int compare_num;
		int total_num;
		bool inited;
		char show_compare;
		search_tool::search_set search_set;
		float anneal;
		float distortion_scale;
		train_set_struct(){
			loss_mod='2';
			decay=0;
			decay_mod='2';
			count=0;
			noise_scale=0;
			data_num=1000;
			T=100;
			total_num=1000000;
			save_freq=1000;
			compare_num=1000;
			inited=false;
			show_compare='2';
			anneal=0.9999;
			distortion_scale=0;
		}
		void load(string path){			
			ifstream fin(path,ios::binary);
			if(!fin)set(path);
			else{
				cout<<endl<<"正在读取训练参数"<<path;
				fin.read((char *)&loss_mod,sizeof(train_set_struct));
				
			}
			show();
		}
		void save(string path){
			cout<<endl<<"正在保存训练参数"<<path;
			ofstream fin(path,ios::binary);
			fin.write((char *)&loss_mod,sizeof(train_set_struct));		
		}
		void show(){
			cout<<endl<<"当前训练参数:";
			cout<<endl<<"\t1.损失函数[1-L1 2-L2 3-L3 c-交叉熵 s-softmax ] :"<<loss_mod;
			cout<<endl<<"\t3.权值衰减系数:"<<decay;
			cout<<endl<<"\t4.权值衰减模式(1-L1 2-L2):"<<decay_mod;
			cout<<endl<<"\t5.噪音比例(与信号的方差的比例):"<<noise_scale;
			cout<<endl<<"\t6.训练样本数"<<data_num;
			cout<<endl<<"\t7.对照样本数"<<compare_num;
			cout<<endl<<"\t8.初始温度"<<T;
			cout<<endl<<"\t9.退温系数"<<anneal;	
			cout<<endl<<"\t0.训练轮次"<<total_num;
			cout<<endl<<"\ta.保存间隔次数"<<save_freq;				
			cout<<endl<<"\tb.显示方式:0.显示计算过程 1.同步计算对照结果 2.不显示计算过程"<<show_compare;
			cout<<endl<<"\te.扭曲程度"<< distortion_scale;			
			
		}
		void set(string path){
			show();
				cout<<endl<<"\tc.确定 l.查看 s.设置最优化方法参数";			
				string s;
				do{
					cout<<endl<<"训练参数>";
					cin>>s;
					switch(s[0]){
					case '1':
						cin>>loss_mod;
						break;
					case '3':
						cin>>decay;
						break;
					case '4':
						cin>>decay_mod;
						break;
					case '5':
						cin>>noise_scale;
						break;
					case '6':
						cin>>data_num;
						break;
					case '7':
						cin>>compare_num;
						break;
					case '8':
						cin>>T;
						break;
					case '9':
						cin>>anneal;
						break;
					case '0':
						cin>>total_num;
						break;
					case 'a':
						cin>>save_freq;
						break;
					case 'b':
						cin>>show_compare;
						break;
					case 'e':
						cin>>distortion_scale;
						break;
					case 'l':
						show();
						cout<<endl<<"\tc.确定 l.查看 s.设置最优化方法参数";			
						break;
					case 's':
						search_set.set();
						break;
					}
				}while(s[0]!='c');
			save(path);
		}
	};
	train_set_struct train_set;
	struct data{
		float **input;
		float **target;
		float *input_data;
		float *target_data;
		float compare_value;
		int data_num;
		int memsize;
		int input_dimen;
		int output_dimen;
		data(){
			input_data=NULL;
			target_data=NULL;
			input=NULL;
			target=NULL;
			data_num=0;
			memsize=0;
			compare_value;
		}
		void malloc(int d_num,int i_dimen,int o_dimen,char train_mod){
			input_dimen=i_dimen;
			output_dimen=o_dimen;
			if(input_data!=NULL){
				memsize-=data_num*input_dimen*sizeof(float);
				cudaFree(input_data);
				CUDA_CHECK;
				if(train_mod!='l'){
					memsize-=data_num*output_dimen*sizeof(float);
					cudaFree(target_data);
					CUDA_CHECK;

				}

				delete [] input;
				delete [] target;
			}
			data_num=d_num;
			cudaMalloc((void**)&input_data,sizeof(float)*d_num*input_dimen);
			memsize+=d_num*input_dimen*sizeof(float);
			CUDA_CHECK;
			if(train_mod!='l'){
				cudaMalloc((void**)&target_data,sizeof(float)*d_num*output_dimen);
				memsize+=d_num*output_dimen*sizeof(float);	
				CUDA_CHECK;
			}
			else target_data=input_data;
			input=new float *[d_num];
			target=new float *[d_num];
			for(int i=0;i<d_num;i++){
				input[i]=input_data+i*input_dimen;
				target[i]=target_data+i*output_dimen;
			}
		}

		void pre(char top_type){
			/*if(top_type=='t'||top_type=='s'||top_type=='r'){
				int len=output_dimen*data_num;
				int blocks=(len+g_threads-1)/g_threads;
				if(top_type=='s')gpu_sigmoid<<<blocks,g_threads>>>(target[0],len);
				else gpu_tanh<<<blocks,g_threads>>>(target[0],len);
				CUDA_CHECK;
			}*/
			compare_value=0;
			float *one;
			cudaMalloc((void**)&one,sizeof(float)*data_num);		
			CUDA_CHECK;
			for(int j=0;j<output_dimen;j++){
				float target_avg=g_sum_value(target[0]+j,data_num,output_dimen);
				target_avg/=-data_num;
				g_set_one(one,data_num);
				CUBT(cublasSscal(cublasHandle,data_num, &target_avg,one, 1));
				float x=1;
				CUBT(cublasSaxpy (cublasHandle,data_num,&x,target[0]+j,output_dimen,one,1));
				CUBT(cublasSnrm2 (cublasHandle,data_num,one, 1,&x));
				x*=x;
				compare_value+=x/data_num;
			}
			cudaFree(one);
			CUDA_CHECK;
		}		
	};	
	struct noise_struct{
		float *scale;
		int dimen;
		void init(int input){
			dimen=input;
			scale=new float[dimen];
		}
	void get_scale(float **data,int data_num){
			float *one;
			float *tmp;
			cudaMalloc((void**)&one,sizeof(float)*data_num);
			CUDA_CHECK;
			cudaMalloc((void**)&tmp,sizeof(float)*data_num);
			CUDA_CHECK;
			g_set_one(one,data_num);
			float avg;
			for(int i=0;i<dimen;i++){
				CUBT(cublasScopy(cublasHandle,data_num,data[0]+i,dimen,tmp,1));
				CUBT(cublasSdot (cublasHandle,data_num, one, 1, tmp,1, &avg));
				avg/=-data_num;			
				CUBT(cublasSaxpy(cublasHandle,data_num,&avg, one, 1,tmp, 1));
				CUBT(cublasSnrm2(cublasHandle,data_num, tmp, 1, scale+i));
				scale[i]/=sqrt((float)data_num);
			}
			cudaFree(one);
			cudaFree(tmp);
		};
		void add(float **data,float **noise_data,int data_num,float noise_scale){
			if(noise_scale==0)return;
			float *rnd;

			CUDA_CHECK;
			curandGenerator_t gen;
			cudaMalloc( (void **) &rnd, (data_num+1)* sizeof(float) ) ;
			CHECK_CURAND( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
			CHECK_CURAND( curandSetPseudoRandomGeneratorSeed(gen,time(NULL)+rand()));
			float tmp=1.0f;
			CUBT(cublasScopy(cublasHandle,data_num*dimen,data[0],1,noise_data[0],1));
			for(int i=0;i<dimen;i++){
				if(scale[i]>0){
					CHECK_CURAND( curandGenerateNormal(gen, rnd,(data_num%2==1)?(data_num+1):data_num,0,scale[i]*noise_scale) );
					CUBT(cublasSaxpy(cublasHandle,data_num,&tmp, rnd, 1,noise_data[0]+i, dimen));
				}
			}
			cudaFree(rnd);
			CUDA_CHECK;
			CHECK_CURAND( curandDestroyGenerator(gen) );


		}

	};
	noise_struct noise;