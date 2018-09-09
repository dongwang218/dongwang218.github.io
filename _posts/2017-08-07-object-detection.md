---
layout: post
mathjax: true
comments: true
title:  "Several Object Detection Papers"
---
This blog is about object detection deep learning papers.

* TOC
{:toc}


# r-cnn

selective search for region proposal, cnn to compute image features, then SVM classify and linear regression to correct bounding box proposal.
details: IoU overlap overlap to determine true label.
fine tuning use proposed region as training examples. SVM training only ground truth box as postive, all other propsoal as negative or ignored.
fine tuning using proposed regions for 200 class training.
Some pool5 unit most active regions: eg 1 unit fires for text.

# SPPNet
main idea: replace last pooling layer by SPP to always get fix length features, before going to fully connected layer. eg SPP 6x6 3x3 2x2 1x1, divide the image into 6x6 no matter how large image is. Pros: can allow any image size, SPP is multilevel thus robust for object deformation. Can speed up detection (one forward for full image, then pick select search region proposal to compute SPP features, then fc to compute features for SVM and bounding box proposal. More accurate classification. Detection accuracy is around mAP 60%, but much faster.

spp is done per filter, still need fully connected layers after.

how to map a region on image to region in feature map. x/product_of_stride map to the feature location.
The best classification is by resize imgae to m x 256, crop.
Training data: standard way is resize to m x 256, crop 224x224. prediction is by 10 view 224 crop.
with SPP: best testing is use the full resize m x 256 image. Training has to use gpu fast, so only two size 224x224 and 180x180 (just resize 224x224).

For detection, only fine tuning the fc layers from imagenet.

tricks: multiple models confidence based (start with different random initialization, need to train conv layers separately).
SPP can improve any network architecture for classification.
corner crops from the entire image can improve classification training.

# fast rcnn
each image and bounding box become one training example. why it is faster is because one image and many roi can be trained in one mini-batch and they share convolutions. Replace imagenet model: last pooling become roi pooling (fixed output size max pooling, instead of fixed input size), then last serveral fc layers replaced by k softmax + 4k coordinates (one set per object class). At predition time, how to deal with multiple proposal's predictions?

rcnn warp the propsoal region and cnn on it, so it is not shared. test time, for each class indepdently, a region is removed if it has IoU overlap with higher scoring regions above a threshold.

incremental paper. interesting one level of SPP can achieve good results when training is one task and conv layers can be retrained too.

one training task includes softmax for classification and per class bounding box regression. replace last max pool by one level of SPP, the size of window is selected to give same # of outputs. Last fc and softmax is replaced with fc+softmax and fc+linear_regressors.
pros: improve accuracy due to multiple objective loss. faster train. test time improve by using svd to replace fc layer.

improved fine tuning: mini-batch includes 2 images, 128 regions per image to share. tune not only the last fc layers (include some conv layers, but not all, definitely not conv1). multi-task loss. label 1 is proposaled region that has >0.5 intersection with truth box.

note: softmax output is k+1, 1 is the background.

bounding box regression is to correct the predict box relative to the truth for each proposaled region. loss is smoothed L1.

training also use pyramid of images, and RoI selects the scale that make it close to 224x224. single scale resize image to m x 600. multiscale does not help.

stagewise svm with hard negative mining? slightly worse than just one pass softmax training.

# faster rcnn

200ms per image
winner of ILSVRC 2015 object detection.
1st place of COCO 2015 object detection competition.

RPN use two conv layesr, it outputs for each location the objectiness score and relative coordidates for 9 reference bounding boxes compared with the true bounding box.

region proposal network (RPN) is a 3x3 conv layer (sliding window), then two sibling 1x1 conv layer (for reg and cls, actually they are fc layer trick). reg layer has 4k outputs, cls has 2k outputs, we have k=9 anchor box at for each 3x3 conv in RPN, with 3 scale and 3 aspect ratios. the proposal is relative to an anchor. num of parameters 512 * (4+2)*9. loss function for cls is whether the anchor has >0.7 overlap with any gound truth box. the bounding box regression loss is the predicted box vs the truth box associated with the anchor when the anchor is positive. training sample 256 anchors per image.

alternating training: start with imagenet model, train RPN first, then fast-rcnn just take proposals from RPN (foreach propsoal compute SPP forward and backward), then iterate to train RPN by fixing conv layers from RCNN.
joint training: difficult to backpro to the RPN's coordinates.

# yolo 2016

Divide image to SxS grid, each grid has B bounding boxes. Using one network to predict whether each bounding box and whether there is an object, also probability of a class. Seems similar to faster-rcnn. But 10 times faster. because rcnn still need to loop through the bounding boxes proposals. 7x7 grid, each has 2 bounding box. Box location and size are all between 0 and 1 a regression problem.

initialization using Glorot&Bengio 2010.

important: yolo score is the iou of predict box and ground truth. S x S x (B * 5 + C). S grid, B num of bbox, C num of categories. 5 is 4 coordinates in terms of grid, last one is iou. output all at the same time. faster-rcnn prduce all proposals, then evaluate one at a time for roi polling. train all outputs by sum of squares (sqrt(width), wegight bbox with object ten time higher). During training, only one bbox out of B is responsible per grid cell for each object (others are ignored).

# SSD

is a refinement of yolo: multiple scale, many more boxes. similar to mtcnn's p-net. reason why it is faster than yolo is because somehow yolo has two fully connected layers instead of all conv layers here.
training objective: bbox only considers default box with > 0.5 overlap with ground truth?, whether there is object or not considers all default boxes. default boxes are the proposals.

important summary: ssd network just produce the concatenated bbox, classes softmax from multiple layers, each layer has a size and aspect ratios. for inference decode_y2 does the nms. For training, the ground truth is encoded in terms of all anchor boxes, so a batch is (batch_size, #boxes, #classes + 4 +4 +4). Bascially based on truth, each box candidate is assigned to one of the classes (background or one class, with the anchor offsets). The loss function now just need to compute the sum of all classification and loalization loss for all boxes, except only top background boxes are identified and considered. So for training, the assignment of which anchor be positive/negative is done per image statically.
Seems quite similar to yolo2, except yolo2 has less layers to predict bbox, and its bbox offset is a different formulation.

# Tensorflow object detection

paper: Speed/accuracy trade-offs for modern convolutional object detectors arXiv:1611.10012v3

expalains r-cnn, r-fcn and ssd well. eg r-fcn is like faster-rcnn, except croping is at the last layers. Fast-rcnn is also anchor based. Anchor should use tiled mutli scale/ratio instead of clustering. ssd append serveral layers on top of feature extractor.

For each feature extractor, pretrained using imagenet, fixed its batch norm parameters. learning rate is tuned and between 2e-4 to 3e-3, and reduce by 10 at different num of steps. Use batch size 1 to handle variable image size.  SSD is trained using rmspro. also decide each layer to extract features, change some to make stride be 16 or 8. SSD is quite similar to RPN of faster-rcnn. did not use fancy crop, just tf's crop_and_resize.

if only one class, then faster-rcnn is like ssd + classifier.


# Retinanet

paper: focal loss for dense object detection

propose focal loss to improve one pass object detection. class imbalance is the culpit for low quality of one pass, because cross entropy loss is still high for easy example, given large num of them (for example large background patches), their sum will overshadow hard examples. focal loss basically add (1-p)^2 to the loss, so easy examples are highly discounted. It uses all 100k backgroudn patches in training. It is also better than online hard-example mining. It does need to initialize sigmoid bias to -log(1-0.01/0.01) to bias prediction to be forground, otherwise training will diverge due to large num of background. use a hourglass shaped feature pyramid network (FPN) as the base, imagenet pretrained resnet. Best coco result eg AP=39.1%.
detail: avg by num of truth box. Guess it is better than hard example mining, because an easy batch can be discounted. I would guess focal loss + hard example mining should be used together.
