train_file_path='E:\ѧϰ\�����\����Ӫ\face\train_face\';
train_img_path_list = dir(strcat(train_file_path,'*.jpg'));
TrainProjectedImages=load('E:\ѧϰ\�����\����Ӫ\face\train_face\TrainProjectedImages.mat');
TrainProjectedImages=TrainProjectedImages.TrainProjectedImages;%ѵ�����ݼ�����ά��ģ� 
m=load('E:\ѧϰ\�����\����Ӫ\face\train_face\m.mat');
m=m.m;%��ֵ����
Eigenfaces=load('E:\ѧϰ\�����\����Ӫ\face\train_face\Eigenfaces.mat');
Eigenfaces=Eigenfaces.Eigenfaces;%ѵ�����ݼ�����ά��ģ� 
% imgTrain=load('D:\Program Files\MATLAB\R2016a\bin\projects\face1\imgTrain.mat');
% imgTrain=imgTrain.imgTrain;%ѵ�����ݼ�����ά��ģ�
[Trainrows,Traincols] = size(TrainProjectedImages);%�õ�ѵ�����Ĵ�С

    img = imread('E:\ѧϰ\�����\����Ӫ\face\test_face\6.jpg'); 
    img2=img;
    img = rgb2gray(img);%תΪ�Ҷ� 
    img=histeq(img);%ֱ��ͼ���⻯
    img=imadjust(img);%ʹ��ͼ���� 1% �����ݱ�������ͺ��������
    img = imresize(img, [64 64], 'nearest');%�ı�ͼ��Ĵ�С��'nearest'��Ĭ��ֵ������ڲ�ֵ
    [irow,icol] = size(img);%�õ�ͼƬ��С
%     for m=1:irow
%         for n=1:icol
%             img(m,n) = 10*img(m,n);
%         end
%     end
    temp = reshape(img,irow*icol,1);%����άͼƬתΪһά����
   
    imgTest = temp;
   
%     imgTrain = imgTrain';
%     mMiu = mean(imgTrain,2);%����еľ�ֵ
%     
%     mMiu = repmat(mMiu,1,icol);%���Ƴ�ԭ�е�����N�еľ���
%     testFace =double(imgTest')-mMiu;
 
   temp = Eigenfaces'*double(imgTest);
   TestProjectedImages = temp;  %�õ� L_eig_vec;
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
img = imread(strcat('E:\ѧϰ\�����\����Ӫ\face\train_face\',train_img_path_list(m).name));
figure('name','best fit')
imshow(img);