function [grad] = calc_gradient(model, input, activations, dv_output)
% Calculate the gradient at each layer, to do this you need dv_output
% determined by your loss function and the activations of each layer.
% The loop of this function will look very similar to the code from
% inference, just looping in reverse.

num_layers = numel(model.layers);
grad = cell(num_layers,1);

% TODO: Determine the gradient at each layer with weights to be updated
for i = num_layers:-1:2
    layer = model.layers(i);
    [~,dv_input,dv_grad] = layer.fwd_fn(activations{i-1},layer.params,layer.hyper_params,true,dv_output);
    grad{i} = dv_grad;
    dv_output = dv_input;
end
layer = model.layers(1);
[~,~,dv_grad] = layer.fwd_fn(input,layer.params,layer.hyper_params,true,dv_output);
grad{1} = dv_grad;