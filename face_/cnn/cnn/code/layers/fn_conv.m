% ----------------------------------------------------------------------
% input: in_height x in_width x num_channels x batch_size
% output: out_height x out_width x num_filters x batch_size
% hyper parameters: (stride, padding for further work)
% params.W: filter_height x filter_width x filter_depth x num_filters
% params.b: num_filters x 1
% dv_output: same as output
% dv_input: same as input
% grad.W: same as params.W
% grad.b: same as params.b
% ----------------------------------------------------------------------

function [output, dv_input, grad] = fn_conv(input, params, hyper_params, backprop, dv_output)  %#ok<INUSL>

[~,~,num_channels,batch_size] = size(input);
[~,~,filter_depth,num_filters] = size(params.W);
assert(filter_depth == num_channels, 'Filter depth does not match number of input channels');

out_height = size(input,1) - size(params.W,1) + 1;
out_width = size(input,2) - size(params.W,2) + 1;
output = zeros(out_height,out_width,num_filters,batch_size);
% TODO: FORWARD CODE
for k = 1:batch_size
    for j = 1:num_filters
        out = conv2(input(:,:,1,k),params.W(:,:,1,j),'valid');
        for i = 2:num_channels
            out = out + conv2(input(:,:,i,k),params.W(:,:,i,j),'valid');
        end
        output(:,:,j,k) = out + params.b(j);
    end
end

dv_input = [];
grad = struct('W',[],'b',[]);

if backprop
	dv_input = zeros(size(input));
	grad.W = zeros(size(params.W));
	grad.b = zeros(size(params.b));
	% TODO: BACKPROP CODE
    for k = 1:batch_size
        for i = 1:num_channels
            for j = 1:num_filters
                output_layer = dv_output(:,:,j,k);
                dv_input(:,:,i,k) = dv_input(:,:,i,k)+ conv2(output_layer,rot90(params.W(:,:,i,j),2),'full');
                grad.W(:,:,i,j) = grad.W(:,:,i,j)+ conv2(rot90(input(:,:,i,k),2),dv_output(:,:,j,k),'valid');
            end
        end
    end    
    for j = 1:num_filters
        grad.b(j) = sum(sum(sum(sum(dv_output(:,:,j,:)))));
    end
end
