# Pytorch Counting Models for Kaggle Sealion Count Challenge

## Overview
With less than two weeks remaining, I decided to jump into the Kaggle Sea Lion count competitiong (https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count) and see if I could get results implementing a few CNN based counting models I'd been reading about. 

Basically an excuse to try Pytorch and  experiment with some new models. Most of my other NN hacking has been in Tensorflow, Torch, or Theano.  

As far as the competition is concerned, these models were a fail. I'm not convinced they couldn't work but I didn't have time to find appropriate hyper parameters, tweak the models, or fix issues in my code to produce anything reasonable. 


I implemented two models
* Count-ception -- From (https://arxiv.org/abs/1703.08710)
* Count-net -- My own mashup of (https://arxiv.org/pdf/1705.10118.pdf) and (https://www.robots.ox.ac.uk/~vgg/publications/2015/Xie15/weidi15.pdf)

I wanted to give the FCRN described in https://www.robots.ox.ac.uk/~vgg/publications/2015/Xie15/weidi15.pdf a shot, but the layer description in the paper between conv4 and FC was vague and I ran out of time.

What's working:
* Kaggle Sealion patch based data processing pipeline with augmentation
* Density or redundant count-ception target generation
* Model training (loss curve looks reasonable) 
* Inference (submission generation)

What's not:
* Good results. Both models train but the appropriate features for Sealion counting and category discrimination do not appear to be learned. Counts are way off. Doing regression across multiple categories of similar looking objects is likely making this a very challenging objective.
* Validation


## Examples

Train:
 
    python train.py /data/sealion/Train-processed/ --batch-size 8 --num-processes 4 --num-gpu 2 --lr 0.001 --opt adadelta --model cc --loss l1

Inference:

    python inference.py /data/x/sealion/Test/ --batch-size 8 --num-processes 4 --restore-checkpoint output/train/20170625-200215/checkpoint-1.pth.tar

 
 