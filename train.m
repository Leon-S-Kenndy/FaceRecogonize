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
    img = rgb2gray(img);%转为灰度 
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
save('.\train_face\TrainProjectedImages.mat','TrainProjectedImages');
save('.\train_face\Eigenfaces.mat','Eigenfaces');
save('.\train_face\m.mat','m');