function [accuracy,loss] = test(model,test_data,test_label,params,numIters) %#ok<INUSD>

num_test = size(test_label,1);
label = zeros(num_test,1);
[output,~] = inference(model,test_data);
for j = 1:num_test
    [~,label(j)] = max(output(:,j));
end

accuracy = sum(label==test_label)/num_test;
loss = loss_crossentropy(output, test_label, [], false);

end

%batch_size = params.batch_size;

%iter = ceil(num_test/batch_size)-1;

%for i = 1:iter
%    it_data = test_data(:,:,:,(i-1)*batch_size+1:i*batch_size);
%    output = inference(model,it_data);
%    for j = 1:batch_size
%        [~,label((i-1)*batch_size+j)] = max(output(:,j));
%    end
%end
%it_data = test_data(:,:,:,iter*batch_size+1:num_test);
%output = inference(model,it_data);
%for j = 1:(num_test-iter*batch_size)
%    [~,label(iter*batch_size+j)] = max(output(:,j));
%end