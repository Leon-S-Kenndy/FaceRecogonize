% ----------------------------------------------------------------------
% input: num_nodes x batch_size
% output: num_nodes x batch_size
% ----------------------------------------------------------------------

function [output, dv_input, grad] = fn_softmax(input, params, hyper_params, backprop, dv_output) %#ok<*INUSD,INUSL>

[num_classes,batch_size] = size(input);
output = zeros(num_classes, batch_size);
% TODO: FORWARD CODE
exp_x = exp(input);
sum_exp = sum(exp_x);
for i = 1:num_classes
    output(i,:) = exp_x(i,:)./sum_exp;
end

dv_input = [];
% This is included to maintain consistency in the return values of layers,
% but there is no gradient to calculate in the softmax layer since there
% are no weights to update.
grad = struct('W',[],'b',[]); 

if backprop
	dv_input = zeros(size(input));
	% TODO: BACKPROP CODE
    for j = 1:batch_size
        dydx = zeros(num_classes, num_classes);
        for i = 1:num_classes
            for k = 1:num_classes
                if (i == k)
                    dydx(i,k) = output(i,j)-output(i,j)*output(i,j);
                else
                    dydx(i,k) = -output(i,j)*output(k,j);
                end
            end
        end
        dv_input(:,j) = dydx*dv_output(:,j);
    end
end


