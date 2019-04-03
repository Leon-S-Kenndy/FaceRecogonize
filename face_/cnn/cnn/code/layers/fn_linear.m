% ----------------------------------------------------------------------
% input: num_in x batch_size
% output: num_out x batch_size
% hyper_params:
% params.W: num_out x num_in
% params.b: num_out x 1
% dv_output: same as output
% dv_input: same as input
% grad: same as params
% ----------------------------------------------------------------------

function [output, dv_input, grad] = fn_linear(input, params, hyper_params, backprop, dv_output)

[num_in,batch_size] = size(input);
assert(num_in == hyper_params.num_in,...
	sprintf('Incorrect number of inputs provided at linear layer.\nGot %d inputs expected %d.',num_in,hyper_params.num_in));

% output = zeros(hyper_params.num_out, batch_size);
% TODO: FORWARD CODE
output = params.W * input;
output = output + params.b;

dv_input = [];
grad = struct('W',[],'b',[]);

if backprop
	% dv_input = zeros(size(input)); 
	% grad.W = zeros(size(params.W));
	% grad.b = zeros(size(params.b));
	% TODO: BACKPROP CODE
    dv_input = params.W.' * dv_output;
    grad.W = dv_output * input.';
    grad.b = sum(dv_output,2);
end
