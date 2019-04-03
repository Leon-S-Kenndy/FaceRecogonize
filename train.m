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
    img = rgb2gray(img);%תΪ�Ҷ� 
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
save('.\train_face\TrainProjectedImages.mat','TrainProjectedImages');
save('.\train_face\Eigenfaces.mat','Eigenfaces');
save('.\train_face\m.mat','m');