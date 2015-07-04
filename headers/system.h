/*配置说明:
（1）VS工程:property ->cuda c/c++ ->Device ->code generation ->sm_**需要是sm_30以上（例如将原来的sm_10改为sm_30或更高)
（2）VS工程:property ->linker ->input ->addition dependencies ->加入 curand.lib;cublas.lib;
（3) 将文件pthreadVC2.dll拷贝到.exe文件的同一目录下
(4)POSIX Threads for Win32的项目，专门为win32开发了一个pthread的li下载到的exe解压之后，会得到三个目录：
其中，Pre-built.2中是已经编译好的lib以及dll，同时包含了一些必要的头文件。将其中的include文件夹和lib文件夹copy到VC的安装目录下
例如，VC6.0的环境，默认安装，则，需要copy到：C:\Program Files\Microsoft Visual Studio\VC98

接着，在编程的时候，引入pthreadVC2.lib即可：

   1: #pragma comment(lib, "pthreadVC2.lib")
 参考：http://www.cnblogs.com/ayanmw/archive/2012/08/06/2625275.html
*/

#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <regex>
#include <sys\stat.h> 
#include<cmath>
#include<cstdlib>
#include<cstring>
#include<cfloat>
#include<ctime>
#include <direct.h> 
#include <sstream>
#include <io.h>
#include <iomanip>  
#include <stdlib.h>
#include <cmath>  
#include <cfloat> 
#include <algorithm>
#include <conio.h>   //监听键盘
#include <WINSOCK2.H>
#pragma comment(lib,"ws2_32.lib")
using namespace std;
#define coutd if(cout_show)cout<<'\n'
//#define coutd cout<<"\n"
#define memzero(x) memset((x),0,_msize((x)))
#define safe_free(x) if((x)!=NULL)delete [] (x)
#define safe_gpu_free(x)  if((x)!=NULL)cudaFree((x))
#define Mrand (rand()%(RAND_MAX+1)*(RAND_MAX+1)+rand())
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h> 
#include <thrust/fill.h>
#include <thrust/replace.h> 
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <curand.h>
#include <pthread.h>
#pragma comment(lib, "pthreadVC2.lib")  //必须加上这句 
using namespace std;
bool cout_show=true;
cublasHandle_t cublasHandle;
int g_threads=-1;
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define CUBT(x) show_cublas_error((x),__LINE__,__FILE__);
#define CUDA_CHECK show_cuda_error(__LINE__,__FILE__);
#define CHECK_CURAND(err) do{\
    if( (err) != CURAND_STATUS_SUCCESS ){\
        fprintf(stderr, "CURAND error %d at file %s line %d.\n", (int)(err), __FILE__, __LINE__);\
  getchar();\
    }}while(0)
void show_cuda_error(int line,char *file){ 
	cudaThreadSynchronize();
    cudaError_t err=cudaGetLastError();//获得最近一次错误的错误代码
   if(err)
	{
		coutd<<"CUDA "<<cudaGetErrorString(err)<<" at line "<<line-1<<"  file "<<file<<endl;;//显示错误内容
		getchar();
	}
}
void show_cublas_error(cublasStatus_t t,int line,char *file){
	cudaThreadSynchronize();
	if(CUBLAS_STATUS_SUCCESS!=t)
	{
		coutd<<"CUBLAS "<<t<<" at line "<<line<<"  file "<<file<<endl;
		getchar();
	}
}
void show_cublas_error(cudaError_t t,int line,char *file){
	cudaThreadSynchronize();
	coutd<<"cublas"<<line;
	coutd<<"CUBLAS "<<t<<"at line "<<line<<"  file "<<file<<endl;
}

void g_gpu_init(){
	if(g_threads!=-1)return;
	 int device_count;
	 if( cudaGetDeviceCount(&device_count) ){
		 coutd<<"【错误】没有发现可用的显卡设备";
		 getchar();
		 getchar();
		 return;
	 }
	/*coutd<<"发现"<<device_count<<"个可用的显卡:";
	for(int i=0;i<device_count;i++){
		 struct cudaDeviceProp device_prop;
		 if(cudaGetDeviceProperties(&device_prop,i)==cudaSuccess){
			 coutd<<"\t"<<i<<"."<<device_prop.name;
		 }
	}*/
	int idx=0;
/*	if(device_count>1){
		do{
		coutd<<"请选则显卡（输入序号）";
		cin>>idx;
		}while(idx<0||idx>=device_count);
	}*/
  	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,idx);
	CUDA_CHECK;
	g_threads=prop.maxThreadsPerBlock;
	//coutd<<"threads per block:"<<g_threads;
	coutd<<"初始化CUBLAS";
	CUBT(cublasCreate(&cublasHandle));
}
void show_gpu_data(float *data,int num,string illu=""){//调试使用
	float *show=new float[num];
	cudaMemcpy(show,data,sizeof(float)*num,cudaMemcpyDeviceToHost);
	CUDA_CHECK;
	coutd<<illu;
	for(int i=0;i<num;i++)cout<<" "<<show[i];
	
}
float gpu_read_unit_value(float *from){
		float ret;
		cudaMemcpy(&ret,from,sizeof(float),cudaMemcpyDeviceToHost);
		CUDA_CHECK;
		return ret;
	}
void gpu_write_unit_value(float *from,float value){
		cudaMemcpy(from,&value,sizeof(float),cudaMemcpyHostToDevice);
		CUDA_CHECK;
	}
void show_cpu_data(float *data,int num=100,string illu=""){//调试使用
	coutd<<illu;
	for(int i=0;i<num;i++)cout<<" "<<data[i];

}

void show_memery_size(int m,char mod='g',string illu=""){
	if(cout_show){
		cout<<illu<<float(m)/1048576<<"M";
		if(mod=='g')cout<<"显存";
		if(mod=='c')cout<<"内存";
	}
}	
struct{
	bool check_folder(string p){
		struct _stat fl;
		string path=p;
		int i=path.size();
		if(path[i-1]=='\\')path[i-1]=0;
		if (!((_stat(path.c_str(), &fl) == 0) && (fl.st_mode & _S_IFDIR)))return 0;
		return 1;
	}
	bool check_file(string p){
		ifstream in(p);  
		if(!in)	return 0;
		in.close();
		return 1;
	}
	void create_folder(string path){
		if(!check_folder(path))mkdir(path.c_str());
	}
	
	bool copy(string ffrom,string fto){
		char ch;
		ifstream in(ffrom,ios::in|ios::binary);             //打开输入文件
		ofstream out(fto,ios::out|ios::binary);              //打开输出文件
		if(!in){
			coutd<<"can not open the file";
			return 0;
		}
		if(!out){
			coutd<<"can not open the file";
			return 0;
		}
		while (in.get(ch))//实现复制功能
			out.put(ch);          

		in.close();
		out.close();
		return 1;
	}
	void show_project_name(string path,string symbol_file_name){
		_finddata_t FileInfo;
		string search =path+"*";
		long Handle = _findfirst(search.c_str(), &FileInfo); 
		if (Handle == -1L){
			coutd<<"默认目录"<<search<<"不存在\n";
			return ;
		}    
		do{
			if (FileInfo.attrib &_A_SUBDIR){
				if( (strcmp(FileInfo.name,".") != 0 ) && (strcmp(FileInfo.name,"..") != 0)){
						string p=path+FileInfo.name+"\\"+  symbol_file_name;
						if(!check_file(p))continue;
						coutd<<"\t<"<<FileInfo.name<<"> ";
				}
			}				
		}while (_findnext(Handle, &FileInfo) == 0); 
		_findclose(Handle);
	}


}file_opt;

struct array_group_sum{
	float *one;
	float alpha;
	float beta;
	int  num;
	int dimen;
	array_group_sum(int dmn,int n){
		dimen=dmn;
		num=n;
		cudaMalloc((void**)&one,sizeof(float)*num);
		thrust :: device_ptr <float > dev_one(one); 
		thrust :: fill ( dev_one , dev_one+num, (float ) 1.0f);
		alpha=1.0f;
		beta=0;
	}
	~array_group_sum(){
		cudaFree(one);
	}
	void sum(float *output,float *array_group){
		CUBT(cublasSgemv(cublasHandle,CUBLAS_OP_N,dimen,num,&alpha,array_group,dimen, one,1,&beta, output,1));
	}
};
template <typename A,typename B>
struct g_array_type_trans {  
__device__  B operator ()( const A & x) const {  
return x;  
}  
}; 
template <typename A,typename B>
void array_type_trans(A *from,B *to,int dimen){//类型转换
	thrust :: device_ptr <A> f(from); 
	thrust :: device_ptr <B> t(to); 
	thrust :: transform (f, f+dimen ,t , g_array_type_trans<A,B>());
}

__global__ void g_array_add_to_matrix(float *w,float *o,float param,int dimen,int data_num){
	int node_idx=blockIdx.x;
	int data_idx=blockIdx.y*blockDim.x+threadIdx.x;
	__shared__ float t;
	if(threadIdx.x==0)t=w[node_idx];
	__syncthreads();
	if(data_idx<data_num){
		o[data_idx*dimen+node_idx]+=t*param;
	}
	
}
void array_add_to_matrix(float *mtx,float *ary,float param,int dimen,int num){//在矩阵中每行加一个向量
	int block_y=(num+g_threads-1)/g_threads;
	dim3 blk(dimen,block_y);
	g_array_add_to_matrix<<<blk,g_threads>>>(ary,mtx,param,dimen,num);		
	CUDA_CHECK;
}
struct g_array_float_plus_double{  
__device__ double operator ()( float & x,double &y) const {  
return x+y;  
}  
}; 
void array_float_plus_double(float *arr,double *dbl,int dimen){//float 与double 相加，结果保留在double中
		thrust::device_ptr<float> a ( arr );
		thrust::device_ptr<double> b (dbl);
		thrust:: transform (a , a+dimen ,b,b,g_array_float_plus_double());
}
float	array_sum(float *arr,int dimen){//矢量各项之和
		thrust::device_ptr<float> a ( arr );
		return  thrust::reduce (a , a+dimen, (float) 0, thrust::plus <float>());
}

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