Web server for FaceCheck IoT project

by Team kawAI


The Facial Recognition engine of FaceCheck is built taking from the following repo:
https://github.com/ZhaoJ9014/face.evoLVe.PyTorch

Specifically, we took his:
1. Pretrained MTCNN and InceptionResnet for generating facial embeddings
2. MTCNN code for facial detection
3. Auxiliary code for face alignment and visualization

Our strategy is to use the pretrained neural networks to generate 512-dimensional
embedding vectors and use the k-nearest neighbors machine learning model to
distinguish embedding vectors for registered individuals.

Our recommendation is to upload at least 3 sample photos per registered individual.
When retraining the kNN model, make sure to use an odd k (1,3,5) and that most individuals
must have at least k sample photos uploaded. Obviously, registered individuals with
no sample photos will not be included when retraining the model.

Download the IRnet (166MB) from the link below then place it in media/models/backbone_cnn.pth
https://s3-ap-southeast-1.amazonaws.com/cs145facecheck/media/models/backbone_cnn.pth


# Notes for collaborators:
* master branch - a working version only in your local machine
* ml+s3 - works with an amazon S3 bucket (all generated and uploaded images go there)
* with_amazon - version without ML models - was deployed successfully with AWS Elastic Beanstalk

Superuser credentials: username - tester1, tester2 / pw - sakura123

If you want to access the S3 bucket, use the credentials in the aws-iam file. Login with the link there, and use the password in there.

When I run this on my machine it consumes 2GB of RAM, clearly because the ML models are memory-heavy (the InceptionResnet serialized file is already 166MB, it will consume more as a loaded model in memory). An amazon EC2 t2.micro instance only has 1GB of ram. I have not yet successfully deployed on an amazon EC2 with the ML model so we can't be sure yet if it really will not work (I have already deployed this FaceCheck django app but without the ML model - see the with_amazon branch). I also tried to deploy on heroku but again memory size problems. So I will look into using SageMaker (AWS free tier available) to integrate with FaceCheck. 

Most probably for the final project version we will be using: 
* EC2 t2.micro compute instance
* S3 storage
* SageMaker notebook server that can service RESTful requests 
* RDS PostgreSQL database (EC2 allows SQLite databases, as  shown in with_amazon branch, but the judges will be impressed with this)
