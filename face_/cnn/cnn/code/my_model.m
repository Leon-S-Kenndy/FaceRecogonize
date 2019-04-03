clear all;
addpath ../data;

it_size = 32;
[train_data,train_label,test_data,test_label]=load_faces(it_size);
train_data = reshape(train_data,it_size,it_size,1,[]);
train_label(train_label == 0) = 20;
test_data = reshape(test_data,it_size,it_size,1,[]);
test_label(test_label == 0) = 20;

addpath layers; % input: 32*32*1
l = [init_layer('conv',struct('filter_size',5,'filter_depth',1,'num_filters',10)) % 28*28*10
    init_layer('relu',[])
	init_layer('pool',struct('filter_size',2,'stride',2)) % 14*14*10
    init_layer('conv',struct('filter_size',5,'filter_depth',10,'num_filters',20)) % 10*10*20
    init_layer('relu',[])
	init_layer('pool',struct('filter_size',2,'stride',2)) % 5*5*20
	init_layer('flatten',struct('num_dims',4))
	init_layer('linear',struct('num_in',500,'num_out',200)) 
    init_layer('relu',[])
    init_layer('linear',struct('num_in',200,'num_out',80))
    init_layer('relu',[])
    init_layer('linear',struct('num_in',80,'num_out',20))
	init_layer('softmax',[])];

model = init_model(l,[it_size it_size 1],20,true);
num_train = size(train_label,1);
num_test = size(test_label,1);
train_loss = [];
test_loss = [];
accuracy = [];
iterate = [];
iteration = 150;
hold on
for i = 1:iteration
   if i <30
       params = struct('learning_rate',double(0.1/i),'weight_decay',0.0005,'batch_size',512,'save_file','mymodel.mat');
   elseif i<100
       params = struct('learning_rate',double(0.05/i),'weight_decay',0.0005,'batch_size',512,'save_file','mymodel.mat');
   else
       params = struct('learning_rate',double(0.005/i),'weight_decay',0.0005,'batch_size',512,'save_file','mymodel.mat');
   end
   
   choose = randsample(num_train,1000);
   train_set_data = train_data(:,:,:,choose);
   train_set_label = train_label(choose);
   [model, loss_train] = train(model,train_set_data,train_set_label,params,1);
   
   choose = randsample(num_test,100);
   test_set_data = test_data(:,:,:,choose);
   test_set_label = test_label(choose);
   [accuracy_test, loss_test] = test(model,test_set_data,test_set_label,params,1);
    
   loss_train 
   loss_test
   accuracy_test
   iterate = [iterate,i];
   train_loss = [train_loss,loss_train]; %#ok<*AGROW>
   test_loss = [test_loss,loss_test];
   accuracy = [accuracy,accuracy_test*100];
   
   if i>=2
        subplot(2,1,1);
        plot(iterate,train_loss,'b',iterate,test_loss,'r');
        legend('train loss','test loss')
        title('Loss');
        drawnow;
        hold on
        subplot(2,1,2);
        plot(iterate,accuracy,'b');
        axis([1 i 0 100]);
%         hold on
%         plot([1 i],[96 96],'r--')
%         axis([1 i 0 100]);
        title('Accuracy');
        drawnow;
        hold on
    end

end



