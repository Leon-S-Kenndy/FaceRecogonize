train_file_path='E:\学习\大二下\科研营\face\train_face\';
train_img_path_list = dir(strcat(train_file_path,'*.jpg'));
TrainProjectedImages=load('E:\学习\大二下\科研营\face\train_face\TrainProjectedImages.mat');
TrainProjectedImages=TrainProjectedImages.TrainProjectedImages;%训练数据集（降维后的） 
m=load('E:\学习\大二下\科研营\face\train_face\m.mat');
m=m.m;%均值人脸
Eigenfaces=load('E:\学习\大二下\科研营\face\train_face\Eigenfaces.mat');
Eigenfaces=Eigenfaces.Eigenfaces;%训练数据集（降维后的） 
% imgTrain=load('D:\Program Files\MATLAB\R2016a\bin\projects\face1\imgTrain.mat');
% imgTrain=imgTrain.imgTrain;%训练数据集（降维后的）
[Trainrows,Traincols] = size(TrainProjectedImages);%得到训练集的大小

    img = imread('E:\学习\大二下\科研营\face\test_face\6.jpg'); 
    img2=img;
    img = rgb2gray(img);%转为灰度 
    img=histeq(img);%直方图均衡化
    img=imadjust(img);%使得图像中 1% 的数据饱和至最低和最高亮度
    img = imresize(img, [64 64], 'nearest');%改变图像的大小，'nearest'（默认值）最近邻插值
    [irow,icol] = size(img);%得到图片大小
%     for m=1:irow
%         for n=1:icol
%             img(m,n) = 10*img(m,n);
%         end
%     end
    temp = reshape(img,irow*icol,1);%将二维图片转为一维向量
   
    imgTest = temp;
   
%     imgTrain = imgTrain';
%     mMiu = mean(imgTrain,2);%求各行的均值
%     
%     mMiu = repmat(mMiu,1,icol);%复制成原有的行数N列的矩阵
%     testFace =double(imgTest')-mMiu;
 
   temp = Eigenfaces'*double(imgTest);
   TestProjectedImages = temp;  %得到 L_eig_vec;
   vDisMin = 9999999999999;
   m=0;
for j=1:Traincols
        mImgTrainCur = TrainProjectedImages(:,j);
        mDis = TestProjectedImages-mImgTrainCur;
        mDis = mDis.^2;
        vDis = sqrt(sum(mDis));
        if vDis<vDisMin
            vDisMin = vDis;
		    m=j;
        end
end
figure('name','original face')
imshow(img2);
fprintf('%d\n',m)
fprintf('best fit:%s\n',train_img_path_list(m).name);
img = imread(strcat('E:\学习\大二下\科研营\face\train_face\',train_img_path_list(m).name));
figure('name','best fit')
imshow(img);