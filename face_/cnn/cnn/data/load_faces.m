function [X_train, Y_train, X_test, Y_test] = load_faces(it_size)

X_tr = [];
X_test = [];
Y_tr = [];
Y_test = [];
summary = csvread('face_label.csv');
for i = 1:length(summary)
% for i = 1:15
    path = ['../data/face/',num2str(summary(i,1)),'.jpg'];
%     path = ['face/',num2str(summary(i,1)),'.jpg'];
    x = imread(path);
    x = imresize(x,[it_size,it_size]);
%     imshow(x);
    x = reshape(x,it_size*it_size,1);
    y = summary(i,2);
    if summary(i,3)==0
        X_tr = [X_tr x];
        Y_tr = [Y_tr y];
    else
        X_test = [X_test x];
        Y_test = [Y_test y];
    end
end

% size(X_tr,2)
perm = randperm(size(X_tr,2));
X_train = X_tr(:,perm);
Y_train = Y_tr(:,perm);

X_train = double(X_train);
X_test = double(X_test);
Y_train = reshape(Y_train, length(Y_train),1);
Y_test = reshape(Y_test, length(Y_test),1);
    
end
