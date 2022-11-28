Deep Learning Mini-Project— Fall 2022

Written by Eduardo Calzadilla

New York University - CS-GY 6953 / ECE-GY 7123

Abstract

Machine Learning techniques have been widely implemented in diverse settings. The implementation of such models has become easier due to the open-sourced nature of their de- velopment, which facilitates transfer-learning. There are in- stances in which available pre-trained networks will yield significantly accurate and generalizable results - needing to change only a few top layer parameters. In other instances certain applications might require programmers to alter pre- mademodelsandtrainfromscratch.However,giventhelarge rangeofparametersavailablefortuning-requiringmorethen a few million can lead to incredibly long training periods that are not viable for productionized uses.

In this paper we will explore different pre-designed models and alter their parameters to achieve a high level of accu- racy ( > 80%) for Image classification susing the CIFAR- 10 Image Dataset. Basing our models on the Train CIFAR10 with PyTorch Github Tutorial ([ link to repo),](https://github.com/kuangliu/pytorch-cifar) we will analyze different pre-designed models and then change some of the establishedparameters to increase accuracy until the desired level.

Given initial results based on 2 epochs the GoogleNet and MobileNetV2 structures were selected. Out of these we con- tinued with the GoogleNet and summarize performance re- sults of altered structure in Figure 3. WE FOUND THE HIGHEST ACCURACY WITH THE FOLLOWING PA- RAMETERS

Introduction

Starting with the pre-designed models found in the Train CI- FAR10 with PyTorch Github Tutorial ([ link to repo)](https://github.com/kuangliu/pytorch-cifar) - and given the accuracy results stated, we chose to do a prelimi- nary test with the following models:

- ResNet18
- GoogleNet
- DenseNet
- MobileNet
- MobileNetV2
- SENet18

Copyright © 2023, Association for the Advancement of Artificial Intelligence (www.aaai.org).![](Aspose.Words.5a1489f3-4af3-44a9-bf45-f840c0f3f521.001.png) All rights reserved.

We then trained each of these for two epochs and compared the training and testing accuracy scores, as well as number of trainable parameters. Out of these, GoogleNet and Mo- bileNetV2 scored the highest overall training and testing scores as well as having the lowest number of parameters (summarized in Figure 1), which gives us more freedom in pulling levers for increased accuracy without exceeding our 5 million parameter limit.

Given initial results based on 2 epochs the GoogleNet and MobileNetV2 structures were selected. Out of these we con- tinuedwiththeGoogleNetgiventhehigherrelativeaccuracy and it’s relative low amount of initial features to work with. We can see their performance summaries in Figure 2 - given the higher performance and capacity for tinker- ing with hyperparameter tuning, we will continue with the GoogleNet architecture for this study.

A brief overview of the GoogleNet architecture as de- signed in [1] Brownee, J. et. al. shows a 22-layer altered CNN. The alteration mostly comes from the introduction of Inception layers, the same which are required to

Copyright

Paper submitted for the CS-GY 6953 / ECE-GY 7123 Deep Learning course for Fall 2022 at New York University. Re- use of this paper and results is permitted gratuitously with appropriate credit.

Methodology

Starting with a higher number of parameters than allowed, we first need to reduce the number of parameters, while leaving all of them without any previous training. Given widespread information on Convolutional Neural Networks (CNNs) we know that decreasing a kernel size will decrease the number of parameters. According to [2] Sood, et al., and others we see that a typically optimal number is 3 (3x3). We verifiedboth with 3 and 5 leaving all else the same, and indeed found 3 yields the least amount of trainable params 6,166,250 vs 6,175,466. Also important to note, is the in- creased accuracy of the larger kernel size, which is to be expected.

While this does not dramatically reduce the number of our parameters for training - we know that the number can, however, be drastically reduced based on the amount of in-

put channels as well as the number of channels manipu- lated throughout the model. We trained the 3 for 20 epochs, and saw no real difference between the 3 in terms of Ac- curacy and Loss, and since MobileNetV2 is an ”improve- ment” on MobileNet and has a lower number of features already present in the model. Therefore, we continued with GoogleNet,toeasilydroptrainableparameterswithoutdras- tically changing performance.

Minimizing the channels led to decreasing the number of parameters by about 3/5. Meaning, the model us now under our constraint of 5 million features and can be continued to have its other hyperparameters altered. However, the result- ing accuracy was far reduced when compared to the original model. In order to keep track of performance - we re-trained and re-tested at every iteration to compare with the original model provided by kuangliu [7].

The initial response was to change kernel sizes to slightly increase complexity in the hopes of increasing image fea- ture detection. After changing the initial kernel sizes, we were still unsatisfied with accuracy scores and looked at changing all intermediate kernel sizes, with the exception of the first 2D Convolution layer in each Inception layer. Since increasing the Kernel size increases complexity if padding is included, we expected an increase in parameters, but not necessarily accuracy. Another strategy was to alter Kernel sizes for the MaxPooling and AvgPooling layers. We decreased the kernel sizes to decrease complexity, but in- creasedpaddingforgreatercontextualawareness,hopingfor higheraccuracywithoutaballooningofparameters[4,6].To further detail the changes we made:

- GoogleNet~~ k5: Intake layer kernel size, changed from 3 to 5
- GoogleNet~~ 0~~ 5: All channels throughout the model re- duced by a factor of 2
- GoogleNet~~ k5~~ 0~~ 5: All channels throughout the model reduced by a factor of 2, and intake layer kernel size was changed from 3 to 5
- GoogleNet~~ k3~~ k5~~ 0~~ 5~~ p2: Changes applied to GoogleNet~~ k5~~ 0~~ 5 remain, and all Inception layer Kernel sizes were changed from 1 to 3 (with the ex- ception of the first Convolution layer in each Inception layer). Padding was increased from 1 to 2, to increase complexity, and/or maintain proper sizes.
- GoogleNet~~ k3~~ k5~~ 0~~ 5~~ p2~~ k5: Changes applied to GoogleNet~~ k3~~ k5~~ 0~~ 5~~ p2 remain, and the Avg Pool layer kernel size was changed from 8 to 5
- GoogleNet~~ 0~~ 5~~ v5: Changes applied to GoogleNet~~ k3~~ k5~~ 0~~ 5~~ p2 remain, the Avg Pool layer Kernel size was changed from 8 to 7 and it’s Padding increased from 1 to 2, and the Max Pool layer Kernel size was changed from 3 to 5

This led to the final two models we decided to continue with - nicknamed ”GoogleNet~~ k5~~ 0~~ 5” and ”GoogleNet~~ 0~~ 5~~ v5” , which then had their learning rate (LR) and weight decay (WD) further altered.

At the end of 8 Epochs for the altered architectures, we did not see accuracy scores at or above 80%. And, given that

it took about 30-45 min for the full training and testing ses- sions of each of these models using a cuda environment in Amazon SageMaker Studio Lab - the same are not suitable for tasks that require training from scratch. To further iterate on these, we considered changing the LR and WD. Increas- ing the former was expected to speed up the epochs, while decreasing the same should increase accuracy [5]. In terms of weight decay however, we expect an increase to lead to less overfittingif present [6].

Results

Choosing the base architecture

When choosing the base architecture, as described in the Methodology section, we ran all available models for just two epochs, evaluating Losses, Accuracy Scores, number of features and time to train in a cuda environment. The results are summarized in Table 1. We can see that only GoogleNet, MobileNet and MobileNetV2 are worth pursuing given the above 4 factors.

![](Aspose.Words.5a1489f3-4af3-44a9-bf45-f840c0f3f521.002.png)

Table 1: Training and test accuracy scores for base models provided by kuangliu

Reduction of Channels and Kernel Size changes

Herewecanindeedseethatwhatwasmosteffectiveforlow- ering trainable parameters was the number of input chan- nels, and overall intermediate channel numbers. When de- creasing the total number of channels throughout the model by a factor of 2 we obtained 1.8 million parameters when compared to the 6.16 million in the original model - as can bee seen on row 3 in Table 2 (corresponding to version ”GoogleNet~~ 0~~ 5”)

Whiletheresultintermsofnumberofparameterswasun- surprising, our accuracy fell further than expected to 68%.

In hopes of increasing accuracy, we looked at ker- nel sizes, and found the scores detailed on Table 2. This shows that after 5 epochs, ”GoogleNet~~ k5~~ 0~~ 5” and ”GoogleNet~~ 0~~ 5~~ v5” had the highest test accuracy scores, with 78.15% and 78.05% respectively. They also maintained a low amount of trainable features.

As expected when we increased Kernel Size we saw an increase in complexity (i.e. parameters) as well as a slight effect in accuracy scores. In order to get even higher accu- racy we tried increasing the padding, and while we found a significant increase in parameters we saw no increase in accuracy, leading us to believe our model was experienc- ing overfitting. To further increase generalizability we in- creased the padding - however found no change to accuracy. Next, we moved to the pooling layers, which significantly simplify our image features. Increasing the Kernel size here surprisingly dropped the accuracy scores by a significant amount(abouta10%drop).Thus,itcouldbeinsteadthatthe padding and kernel sizes are increasing the complexity but

![](Aspose.Words.5a1489f3-4af3-44a9-bf45-f840c0f3f521.003.png)

Table 2: Training and test accuracy scores for models based on the GoogleNet architecture.

essentially blurring out the visual context, rather than pro- viding some. Some researchers have shown that a 3x3 pool- ing layer is usually the best for increasing training accuracy and maintaining generalizability [7,8]. Then, decreasing the Avg Pool layer Kernel size to 5 we found an increase in ac- curacy, to further alter the same we then increased the Max Pool layer Kernel size to further increase testing accuracy.

Overall ”GoogleNet~~ k5~~ 0~~ 5” and ”GoogleNet~~ 0~~ 5~~ v5” had the highest test accuracy scores, while having flexibil- ity in number of features (if further structural iterations are necessary).

Training Hyperparameter Tuning

The the performance of the selected models was then veri- fiedwith different learning parameters. In order to deal with the expected overfitting to our training data - we tinkered with the LR and WD, the results are shown in Table 3.

![](Aspose.Words.5a1489f3-4af3-44a9-bf45-f840c0f3f521.004.png)

Table 3: Training and test accuracy scores for mod- els given different learning parameters, based on the al- tered GoogleNet architectures - ”GoogleNet~~ k5~~ 0~~ 5” and ”GoogleNet~~ 0~~ 5~~ v5”.

As expected an increase in the LR led to overall less ac- curacy, with ”GoogleNet~~ 0~~ 5~~ v5” appearing more suscep- tible to a change in LR (Figure 1b). Decreasing the LR by half, greatly increases training accuracy and testing accu- racy.Whilewecouldstophere,weincludedsomeadditional changes for comparison.

Since there still could be some degree of overfitting, which could balloon in other datasets, we experimented with the WD. We saw that an increase in WD greatly de- creases accuracy scores, particularly when related to testing accuracy. It also seems the ”GoogleNet~~ k5~~ 0~~ 5” model was much more susceptible to a WD change (Figure 1a).

![](Aspose.Words.5a1489f3-4af3-44a9-bf45-f840c0f3f521.005.png)

Figure 1a: GoogleNet~~ k5~~ 0~~ 5 scores

![](Aspose.Words.5a1489f3-4af3-44a9-bf45-f840c0f3f521.006.png)

Figure 1b: GoogleNet~~ 0~~ 5~~ v5 scores

Conclusion

We found the highest accuracy scores for both train- ing and testing with the altered GoogleNet models: ”GoogleNet~~ k5~~ 0~~ 5” and ”GoogleNet~~ 0~~ 5~~ v5”, set with a LR = 0.05; and WD = 5E-5. The Test results being 85.01% and 86.31% and each took about 25 min to complete in a cuda environment. This should satisfy our needs for ad-hoc training and testing with limited computing resources. How- ever, to further understand one of our models, we trained ”GoogleNet~~ 0~~ 5~~ v5” for 25 epochs and found that after the 11th Epoch the accuracy scores start to converge at 93% for training and 88% for testing.

Appendix

Please findthe model architecture in the following Git Repo ([ link to repo)](https://github.com/EdAlexc/nyu-mini-project)

References

1. Brownlee, J. (2020, September 11). Under- stand the impact of learning rate on neural network performance. MachineLearningMas- tery.com. Retrieved November 27, 2022, from https://machinelearningmastery.com/understand-the- dynamics-of-learning-rate-on-deep-learning-neural- networks/: :text=The
1. Shivani Sood, Harjeet Singh (2022). Effect of Ker- nel Size in Deep Learning-Based Convolutional Neural

Networks for Image Classification. ECS Transactions, 107(1), 8877.

3. Shaikh, J. F. (2020, May 14). Inception network: Implementation of googlenet in Keras. Analyt- ics Vidhya. Retrieved November 27, 2022, from ’https://www.analyticsvidhya.com/blog/2018/10/ understanding-inception-network-from-scratch/’
3. CIFAR-10 examples¶. CIFAR-10 examples - An open source AutoML toolkit for neural architecture search, model compression and hyper-parameter tuning (NNI v2.0). (n.d.). Retrieved November 27, 2022, from https://nni.readthedocs.io/en/v2.0/TrialExample/ Cifar10Examples.html
3. Brownlee, J. (2020, September 11). Under- stand the impact of learning rate on neural network performance. MachineLearningMas- tery.com. Retrieved November 27, 2022, from https://machinelearningmastery.com/understand-the- dynamics-of-learning-rate-on-deep-learning-neural- networks/: :text=The
3. Zhang, A., Lipton, Z. C., Li, M., amp; Smola, A. J. (2020). Chapter 3: linear Neural Networks for Regres- sion. In Dive into deep learning release 0.14.3. essay. associated website: https://d2l.ai/index.html
3. Kuangliu. (n.d.). Kuangliu/Pytorch-CIFAR: 95.47
