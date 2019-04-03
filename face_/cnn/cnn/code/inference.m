function [output,activations] = inference(model,input)
% Do forward propagation through the network to get the activation
% at each layer, and the final output

num_layers = numel(model.layers);
activations = cell(num_layers,1);

% TODO: FORWARD PROPAGATION CODE
y = input;
for i = 1:num_layers
    x = y;
    layer = model.layers(i);
    [y,~,~] = layer.fwd_fn(x,layer.params,layer.hyper_params,false,[]);
    activations{i} = y;
end

output = activations{end};
