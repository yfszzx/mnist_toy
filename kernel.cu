#define DATA_CLASS mnist_data_set 
#define ML_CLASS main_train
#include "headers/system.h"
#include "headers/search_tool.h"
#include "headers/pca.h"
#include "data_set/mnist.h"
#include "headers/perceptrons.h"
#include "headers/multi_perceptrons.h"
#include "headers/data_set.h"
#include "headers/train.h"
#include "headers/bagging.h"
#include "data_set/mnist_main.h"

void main(){
	//multi_perceptrons_sample *s=new multi_perceptrons_sample("");//�㷨����
	mnist_main.main();

}



