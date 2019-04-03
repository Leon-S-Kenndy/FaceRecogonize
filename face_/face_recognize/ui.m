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
function pushbutton1_Callback(hObject, eventdata, handles)%ѵ��
file_path='.\train\';% ͼ���ļ���·�� 
img_path_list = dir(strcat(file_path,'*.jpg'));%��ȡ���ļ���������jpg��ʽ��ͼ��
img_num = length(img_path_list);%��ȡͼ��������
imgTrain = [];
Q = [];%�о���һ��ͼ��
trainFace = [];%��ά���ѵ�������ľ���
% ��ȡÿһ��ͼ��
%ת��Ϊ�Ҷ�ͼ�� ����ÿһ��ͼ��ת��Ϊ������ Ȼ��ϲ�Ϊ����T
for j=1:img_num %��һ��ȡͼ��
    image_name = img_path_list(j).name;%ͼ����
    img = imread(strcat(file_path,image_name));
%     img = rgb2gray(img);%תΪ�Ҷ� 
    img=histeq(img);%ֱ��ͼ���⻯
    img=imadjust(img);%ʹ��ͼ���� 1% �����ݱ�������ͺ��������
    img = imresize(img, [64 64], 'nearest');
    [irow,icol] = size(img);%�õ�ͼƬ��С
%     for m=1:irow
%         for n=1:icol
%             img(m,n) = 10*img(m,n);
%         end
%     end
    temp = reshape(img,irow*icol,1);%����άͼƬתΪһά����
    Q = [Q,temp]; % ÿ��ͼƬ����Ϣ��ΪV��һ��   
    imgTrain=[imgTrain,temp]; 
end
 
%  [L_eig_vec��A] = HSCalPCA(imgTrain);%����PCA��ά�������õ�ӳ�����
%�õ���ֵ����
%����T��ȥ��ֵ����  �õ�����A
m = mean(imgTrain,2); % ƽ��ͼ��/��ƽ����ÿһ��ͼ��Ķ�Ӧ������ƽ����m=(1/P)*sum(Tj's) (j=1 : P)
Train_Number = size(imgTrain,2);%����
%�����ÿһ��ͼƬ����ֵͼ��ķ���
A = [];  
for i = 1 : Train_Number%��ÿһ��
    temp = double(imgTrain(:,i))-m; %ÿһ��ͼ���ֵ�Ĳ���
    A = [A temp]; %�����
end
%�õ�A��Э�������ת�õõ�L
L = A'*A; % L��Э�������C=A*A'��ת��
%�õ�����ֵ����������
[V D] = eig(L); %�Խ����ϵ�Ԫ����L|C������ֵ.V:����������Ϊ�е����Ⱦ���D������ֵ�ԽǾ��󡣼�L*V = V*D.
L_eig_vec = [];%����ֵ����
max=0;
for i = 1 : size(V,2)%��ÿ����������   
   max=max+D(i,i);
end
sum=0;

for i = size(V,2):-1:1%��ÿ����������      
    L_eig_vec = [L_eig_vec V(:,i)];%���ж�Ӧ����������
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
 
Eigenfaces = A * L_eig_vec; % �����Э�������C������������
                            %�õ���ά�˵�����,AΪÿһ��ͼ�����ֵͼ��ķ���ɵľ���
TrainProjectedImages = [];%ӳ��ͼ��
for i = 1 : img_num%����ÿһ��ѵ������
    temp = Eigenfaces'*double(Q(:,i));
    TrainProjectedImages = [TrainProjectedImages temp];  %�õ� L_eig_vec;
end
save('.\train\TrainProjectedImages.mat','TrainProjectedImages');
save('.\train\Eigenfaces.mat','Eigenfaces');
save('.\train\m.mat','m');
msgbox('ѵ�����', 'ѵ��');
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)%׼ȷ�ʲ鿴
train_file_path='.\train\';
train_img_path_list = dir(strcat(train_file_path,'*.jpg'));
TrainProjectedImages=load('.\train\TrainProjectedImages.mat');
TrainProjectedImages=TrainProjectedImages.TrainProjectedImages;%ѵ�����ݼ�����ά��ģ� 
m=load('.\train\m.mat');
m=m.m;%��ֵ����
a=xlsread('label_all.csv');%��ȡ��ǩ
Eigenfaces=load('.\train\Eigenfaces.mat');
Eigenfaces=Eigenfaces.Eigenfaces;%ѵ�����ݼ�����ά��ģ� 
% imgTrain=load('D:\Program Files\MATLAB\R2016a\bin\projects\face1\imgTrain.mat');
% imgTrain=imgTrain.imgTrain;%ѵ�����ݼ�����ά��ģ�
[Trainrows,Traincols] = size(TrainProjectedImages);%�õ�ѵ�����Ĵ�С

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
        img=histeq(img);%ֱ��ͼ���⻯
        img=imadjust(img);%ʹ��ͼ���� 1% �����ݱ�������ͺ��������
        img = imresize(img, [64 64], 'nearest');%�ı�ͼ��Ĵ�С��'nearest'��Ĭ��ֵ������ڲ�ֵ
        [irow,icol] = size(img);%�õ�ͼƬ��С
        temp = reshape(img,irow*icol,1);%����άͼƬתΪһά����
        imgTest = temp;
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
msgbox(acc,'׼ȷ��');

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
[FileName,PathName,FilterIndex]=uigetfile('.jpg','ѡ�����ͼ');
str=[PathName FileName];

train_file_path='.\train\';
train_img_path_list = dir(strcat(train_file_path,'*.jpg'));
TrainProjectedImages=load('.\train\TrainProjectedImages.mat');
TrainProjectedImages=TrainProjectedImages.TrainProjectedImages;%ѵ�����ݼ�����ά��ģ� 
m=load('.\train\m.mat');
m=m.m;%��ֵ����
Eigenfaces=load('.\train\Eigenfaces.mat');
Eigenfaces=Eigenfaces.Eigenfaces;%ѵ�����ݼ�����ά��ģ� 
% imgTrain=load('D:\Program Files\MATLAB\R2016a\bin\projects\face1\imgTrain.mat');
% imgTrain=imgTrain.imgTrain;%ѵ�����ݼ�����ά��ģ�
[Trainrows,Traincols] = size(TrainProjectedImages);%�õ�ѵ�����Ĵ�С

    img = imread(str); 
    img2=img;
%     img = rgb2gray(img);%תΪ�Ҷ� 
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
% figure('name','original face')-

subplot(2,1,1);
imshow(img2);
title('ԭͼ')
fprintf('%d\n',m)


img = imread(strcat('.\train\',train_img_path_list(m).name));
% figure('name','best fit')
subplot(2,1,2);
imshow(img);
title('���ͼ')
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
