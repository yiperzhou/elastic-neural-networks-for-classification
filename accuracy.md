# classification on **CIFAR-10**

model                                  | error               |     FLOPs                                | Total params      
---------------------------------------| ------------------- | ---------------------------------------- | ---------------------- 
MobileNet                              | 10.3700             |  311,969,280                             |    2,209,306           
Elastic-MobileNet                       | 7.8200             |  312,013,920                             |   2,254,076            
InceptionV3                            | 6.3900              |  2,836,385,120                           |     22,314,794         
Elastic_InceptionV3                    | 4.4800              |  2,836,485,600                           |     22,415,384          
ResNet-50                               | 5.5400              | 3,832,303,616                            |  23,608,202          
Elastic_ResNet-50                       | 6.1900              |  3,832,454,656                           |      23,759,402      
ResNet-152                              |     4.38            |     11,256,745,984                       |   58,467,146    
Elastic_ResNet-152-weight-Depth         |   4.9600            |         11,257,204,224                    |    58,925,876    
Elastic_ResNet-152-weight-1             | 4.5100              |     11,257,204,224                       |    58,925,876 

# classification on **CIFAR-100**

model                                  | error               |       FLOPs                             | Total params       
---------------------------------------| ------------------- | ----------------------------------------| ----------------- 
MobileNet                              | 38.6000             | 315,356,160                             |  5,596,276         
Elastic-MobileNet                      | 27.8200             | 315,802,560                             | 6,043,976        
InceptionV3                            | 24.9800             | 2,840,993,120                           | 26,922,884        
Elastic_InceptionV3                    |  22.3400            | 2,841,997,920                           | 27,928,784           
ResNet-50                               | 21.9600             |    3,832,487,936                        | 23,792,612        
Elastic_ResNet-50                       | 24.2000            |  3,833,998,336                          | 25,304,612       
ResNet-152                             |     19.03           |     11,256,930,304                         |  58,651,556     
Elastic_ResNet-152-weight-Depth         | 20.3900            |    11,261,512,704                         |  63,238,856          
Elastic_ResNet-152-weight-1             |  20.0600           |        11,261,512,704                       |    63,238,856

* [1] references second-to-last layer output classifier best accuracy  
* [2] references last layer(final layer) output classifier best accuracy  
* [3] FLOPs references **Total float point operations**, A multiply-add counts as one flop  
* for all above elastic-models, include **all** intermediate layers outputs  

* 添加 model loss 或者val loss，而不需要f1 score-要了解清楚为什么不能要f1 score，要和heikki解释清楚


# classification accuracy on **CIFAR 10** 

model                                  | error               |     FLOPs                                | Total params      
---------------------------------------| ------------------- | ---------------------------------------- | ---------------------- 
ResNet-152                             |     4.38            |     11,256,745,984                       |   58,467,146    
Elastic-ResNet-weight-Depth[1]         |   4.9600            |         11,257,204,224                    |    58,925,876    
Elastic-ResNet-weight-1[2]             | 4.5100              |     11,257,204,224                       |    58,925,876 


# classification accuracy on **CIFAR 100** 

model                                  | error               |     FLOPs                                | Total params      
---------------------------------------| ------------------- | ---------------------------------------- | ---------------------- 
ResNet-152                             |     19.03           |     11,256,930,304                         |  58,651,556     
Elastic-ResNet-152-weight-Depth         | 20.3900            |    11,261,512,704                         |  63,238,856          
Elastic-ResNet-152-weight-1             |  20.0600           |        11,261,512,704                       |    63,238,856


# classification accuracy on **CIFAR-10**  

model                                  | error (best)               |     FLOPs [3]                               | Total params      
---------------------------------------| ------------------- | ---------------------------------------- | ---------------------- 
InceptionV3                            | 6.3900              |  2,836,385,120                           |     22,314,794         
Elastic-InceptionV3                    | 4.3100[1], 4.5300 [2]|  2,836,485,600                           |     22,415,384          


# classification accuracy on **CIFAR-100**  

model                                  | error (best)               |       FLOPs                             | Total params       
---------------------------------------| ------------------- | ----------------------------------------| ----------------- 
InceptionV3                            | 24.9800             | 2,840,993,120                           | 26,922,884        
Elastic-InceptionV3                    |  22.3400            | 2,841,997,920                           | 27,928,784           

* [1] references second-to-last layer output classifier best accuracy
* [2] references last layer(final layer) output classifier best accuracy
* [3] FLOPs references **Total float point operations**, A multiply-add counts as one flop


# classification accuracy on **CIFAR-10**  

model                                  | error (best)               |     FLOPs [3]                               | Total params      
---------------------------------------| ------------------- | ---------------------------------------- | ---------------------- 
MobileNet-alpha-0.75                  | 10.3700             |  311,969,280                             |    2,209,306           
Elastic-MobileNet-alpha-0.75          | 7.8200[1], 8.1000[2]|  312,013,920                             |   2,254,076            


# classification accuracy on **CIFAR-100**  

model                                  | error (best)               |       FLOPs                             | Total params       
---------------------------------------| ------------------- | ----------------------------------------| ----------------- 
MobileNets-alpha-0.75                  | 38.6000             | 315,356,160                             |  5,596,276         
Elastic-MobileNets-alpha-0.75          | 27.8200, 28.8100    | 315,802,560                             | 6,043,976        

* [1] references second-to-last intermediate layer output classifier best accuracy
* [2] references last layer(final layer) output classifier best accuracy
* [3] FLOPs references **Total float point operations**, A multiply-add counts as one flop



# classification accuracy on **CIFAR-10**  

model                                  | error (best)               |     FLOPs [3]                               | Total params      
---------------------------------------| ------------------- | ---------------------------------------- | ---------------------- 
ResNet50                               | 5.5400              | 3,832,303,616                            |  23,608,202          
Elastic-ResNet50                       | 6.1900              |  3,832,454,656                           |      23,759,402      


# classification accuracy on **CIFAR-100**  

model                                  | error (best)               |       FLOPs [3]                            | Total params       
---------------------------------------| ------------------- | ----------------------------------------| ----------------- 
ResNet50                               | 21.9600             |    3,832,487,936                        | 23,792,612        
Elastic-ResNet50                       | 24.2000[1]，24.3900[2]    |  3,833,998,336                          | 25,304,612       

* [1] references second-to-last intermediate layer output classifier best accuracy
* [2] references last layer(final layer) output classifier best accuracy
* [3] FLOPs references **Total float point operations**, A multiply-add counts as one flop

reference
1. Huang, G., Liu, Z., Weinberger, K.Q. and van der Maaten, L., 2017, July. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (Vol. 1, No. 2, p. 3).
http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf

2. 
* FLOPs calculation, Supported layers are Conv1D/2D/3D and Dense; not including BatchNormalization, Add, Activation, DepthwiseConv2D,  

** 注意，注意用的flops是用于101类的分类，而在我的实验中，分别是10和100， final_output = Dense(101, activation='sigmoid', name='final_output')(w)， 这是不一样的 **


Note
1. in 101 multi-classes ResNet in Yue's experiment, its Total params: 25,321,781 but in our experiment, which number of multiclass is 100, the total params is 23,792,612， these two numbers are different, since the number of class is different, here is the comparison detail.


Elastic-ResNet with num_class = 100   | Elastic-ResNet with num_class = 101
------------------------------| ---------------------------  
cifar 100                     |      age estimation  
Total params: 25,304,612           |   Total params: 25,321,781
Trainable params: 1,716,900          |   Trainable params: 1,734,069
Non-trainable params: 23,587,712   |   Non-trainable params: 23,587,712

* 这里的可训练参数不一样是在 intermediate_add_1, 2, 3, 4 这些中间层append到网络模型的最后时的参数数量不一样

2. 这里应该不能直接用白月之前计算好的flops，因为在log.txt中，intermediate_add_* 也有flop数，根据params的计算数可知，flops也是和num_class的数量有关的，而白月的实验中num_class = 101, 而我的实验中num_class=100
3. A multiply-add counts as one flop
4. ElasticNN-InceptionV3 was run with only adding intermediate layers [8, 9, 10, 11], so all previous results are based on 4 intermeidate layers.



