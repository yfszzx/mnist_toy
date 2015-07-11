//#define STK_MOD
#define TEST_MOD
#ifdef STK_MOD
#define DATA_CLASS stock_set
#else
#define DATA_CLASS mnist_data_set
#endif
#define ML_CLASS main_train
#include "headers/system.h"
#include "headers/search_tool.h"
#include "headers/pca.h"
#include "data_set/mnist.h"
#include "data_set/stock.h"
#include "headers/perceptrons.h"
#include "headers/multi_perceptrons.h"
#include "headers/data_set.h"
#include "headers/train.h"
#include "headers/deep_train.h"
#include "headers/bagging.h"
#include "data_set/mnist_main.h"
#include "data_set/stock_main.h"
void main(){
	//multi_perceptrons_sample *s=new multi_perceptrons_sample("");//À„∑®≤‚ ‘
#ifdef STK_MOD
stock_main.main();
#else
mnist_main.main();
#endif
}





