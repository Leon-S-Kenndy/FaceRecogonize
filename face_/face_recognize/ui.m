function varargout = ui(varargin)
% UI MATLAB code for ui.fig
%      UI, by itself, creates a new UI or raises the existing
%      singleton*.
%
%      H = UI returns the handle to a new UI or the handle to
%      the existing singleton*.
%
%      UI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in UI.M with the given input arguments.
%
%      UI('Property','Value',...) creates a new UI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before ui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to ui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help ui

% Last Modified by GUIDE v2.5 03-Aug-2018 23:10:30

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ui_OpeningFcn, ...
                   'gui_OutputFcn',  @ui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before ui is made visible.
function ui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to ui (see VARARGIN)

% Choose default command line output for ui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes ui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = ui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)%训练
file_path='.\train\';% 图像文件夹路径 
img_path_list = dir(strcat(file_path,'*.jpg'));%获取该文件夹中所有jpg格式的图像
img_num = length(img_path_list);%获取图像总数量
imgTrain = [];
Q = [];%列矩阵，一副图像
trainFace = [];%降维后的训练样本的矩阵
% 读取每一幅图像
%转化为灰度图像 并将每一幅图像转化为列向量 然后合并为矩阵T
for j=1:img_num %逐一读取图像
    image_name = img_path_list(j).name;%图像名
    img = imread(strcat(file_path,image_name));
%     img = rgb2gray(img);%转为灰度 
    img=histeq(img);%直方图均衡化
    img=imadjust(img);%使得图像中 1% 的数据饱和至最低和最高亮度
    img = imresize(img, [64 64], 'nearest');
    [irow,icol] = size(img);%得到图片大小
%     for m=1:irow
%         for n=1:icol
%             img(m,n) = 10*img(m,n);
%         end
%     end
    temp = reshape(img,irow*icol,1);%将二维图片转为一维向量
    Q = [Q,temp]; % 每张图片的信息做为V的一列   
    imgTrain=[imgTrain,temp]; 
end
 
%  [L_eig_vec，A] = HSCalPCA(imgTrain);%调用PCA降维函数，得到映射矩阵
%得到均值矩阵
%并用T剪去均值矩阵  得到矩阵A
m = mean(imgTrain,2); % 平均图像/行平均（每一副图像的对应象素求平均）m=(1/P)*sum(Tj's) (j=1 : P)
Train_Number = size(imgTrain,2);%列数
%计算机每一张图片到均值图像的方差
A = [];  
for i = 1 : Train_Number%对每一列
    temp = double(imgTrain(:,i))-m; %每一张图与均值的差异
    A = [A temp]; %差矩阵
end
%得到A的协方差矩阵并转置得到L
L = A'*A; % L是协方差矩阵C=A*A'的转置
%得到特征值与特征向量
[V D] = eig(L); %对角线上的元素是L|C的特征值.V:以特征向量为列的满秩矩阵，D：特征值对角矩阵。即L*V = V*D.
L_eig_vec = [];%特征值向量
max=0;
for i = 1 : size(V,2)%对每个特征向量   
   max=max+D(i,i);
end
sum=0;

for i = size(V,2):-1:1%对每个特征向量      
    L_eig_vec = [L_eig_vec V(:,i)];%集中对应的特征向量
    sum=sum+D(i,i);
    if(sum/max>0.99)
        break;
    end
end
 
% for i = 1:size(V,2)
%      if(D(i,i)>1)
% L_eig_vec = [L_eig_vec V(:,i)];
%     end
% end
 
Eigenfaces = A * L_eig_vec; % 计算机协方差矩阵C的特征向量，
                            %得到降维了的特征,A为每一张图像与均值图像的方差构成的矩阵，
TrainProjectedImages = [];%映射图像
for i = 1 : img_num%对于每一个训练特征
    temp = Eigenfaces'*double(Q(:,i));
    TrainProjectedImages = [TrainProjectedImages temp];  %得到 L_eig_vec;
end
save('.\train\TrainProjectedImages.mat','TrainProjectedImages');
save('.\train\Eigenfaces.mat','Eigenfaces');
save('.\train\m.mat','m');
msgbox('训练完成', '训练');
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)%准确率查看
train_file_path='.\train\';
train_img_path_list = dir(strcat(train_file_path,'*.jpg'));
TrainProjectedImages=load('.\train\TrainProjectedImages.mat');
TrainProjectedImages=TrainProjectedImages.TrainProjectedImages;%训练数据集（降维后的） 
m=load('.\train\m.mat');
m=m.m;%均值人脸
a=xlsread('label_all.csv');%获取标签
Eigenfaces=load('.\train\Eigenfaces.mat');
Eigenfaces=Eigenfaces.Eigenfaces;%训练数据集（降维后的） 
% imgTrain=load('D:\Program Files\MATLAB\R2016a\bin\projects\face1\imgTrain.mat');
% imgTrain=imgTrain.imgTrain;%训练数据集（降维后的）
[Trainrows,Traincols] = size(TrainProjectedImages);%得到训练集的大小

    dirOutput=dir('.\test\*.jpg');
    count=length(dirOutput);
    fileNames={dirOutput.name};
    right=0;
    for i=1:count
        num=num2str(10*(i-1));
        f=strcat(num,'.jpg');
        true=a(1+10*(i-1),2);
        img = imread(strcat('.\test\',f));
        img2=img;
        img=histeq(img);%直方图均衡化
        img=imadjust(img);%使得图像中 1% 的数据饱和至最低和最高亮度
        img = imresize(img, [64 64], 'nearest');%改变图像的大小，'nearest'（默认值）最近邻插值
        [irow,icol] = size(img);%得到图片大小
        temp = reshape(img,irow*icol,1);%将二维图片转为一维向量
        imgTest = temp;
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
        fprintf('%d\n',m)
        fprintf('best fit:%s\n',train_img_path_list(m).name);
        test=train_img_path_list(m).name;
        test=test(1:end-4);
        test=str2num(test);
        test=a(test+1,2);
        img = imread(strcat('.\train\',train_img_path_list(m).name));
        if abs(test-true)<0.0000001
            right=right+1;
        end
    end
acc=right/i
acc=num2str(acc);
msgbox(acc,'准确率');

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
[FileName,PathName,FilterIndex]=uigetfile('.jpg','选择测试图');
str=[PathName FileName];

train_file_path='.\train\';
train_img_path_list = dir(strcat(train_file_path,'*.jpg'));
TrainProjectedImages=load('.\train\TrainProjectedImages.mat');
TrainProjectedImages=TrainProjectedImages.TrainProjectedImages;%训练数据集（降维后的） 
m=load('.\train\m.mat');
m=m.m;%均值人脸
Eigenfaces=load('.\train\Eigenfaces.mat');
Eigenfaces=Eigenfaces.Eigenfaces;%训练数据集（降维后的） 
% imgTrain=load('D:\Program Files\MATLAB\R2016a\bin\projects\face1\imgTrain.mat');
% imgTrain=imgTrain.imgTrain;%训练数据集（降维后的）
[Trainrows,Traincols] = size(TrainProjectedImages);%得到训练集的大小

    img = imread(str); 
    img2=img;
%     img = rgb2gray(img);%转为灰度 
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
% figure('name','original face')-

subplot(2,1,1);
imshow(img2);
title('原图')
fprintf('%d\n',m)


img = imread(strcat('.\train\',train_img_path_list(m).name));
% figure('name','best fit')
subplot(2,1,2);
imshow(img);
title('最佳图')
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
