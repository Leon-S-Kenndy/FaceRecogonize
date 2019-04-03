% ----------------------------------------------------------------------
% input: num_nodes x batch_size
% labels: batch_size x 1
% ----------------------------------------------------------------------

function [loss, dv_input] = loss_crossentropy(input, labels, hyper_params, backprop) %#ok<INUSL>

assert(max(labels) <= size(input,1));

% TODO: CALCULATE LOSS
label = zeros(size(input));
[~,batch_size] = size(input);
for i = 1:batch_size
    label(labels(i),i) = 1;
end
loss = -sum(sum(label.*log(input)))/batch_size();

dv_input = zeros(size(input));
if backprop
	% TODO: BACKPROP CODE
    dv_input = -label.*(1./input)/batch_size;
end
