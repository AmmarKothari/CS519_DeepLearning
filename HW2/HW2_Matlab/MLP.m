
function MLP()
data_points = 1000;
input_dims = 10;

%and function data
x = randi([0,1],[data_points,input_dims]);
y = sum(x,2) <= input_dims/2;

% %XOR function
% x = randi([0,1],[data_points,2]);
% y = x(:,1) ~= x(:,2);


hidden_nodes = 5;
bias = 1;
x_b = [x, bias*ones(data_points, 1)];
x_shape = size(x_b);
init_c = 1;
init_spread = 2;
W1 = init_c - init_spread*rand(hidden_nodes, x_shape(2));
W2 = init_c - init_spread*rand(hidden_nodes + 1, 1);
learning_rate = 2.6;

L = 0;
pred = 0;
NL2_hist = [];
accuracy = [];
acc_p = [];
dL_w2_hist = [];
dL_w1_hist = [];
L_hist = [];
for i2 = 1:500
    acc_local = [];
    dL_w2 = 0;
    dL_w1 = 0;
    for i1 = 1:length(x_b)
        L1  = W1*x_b(i1, :)';
        NL1 = 1./(1 + exp(-L1));
        NL1_b = [NL1; bias]';
        L2 = NL1_b * W2;
        NL2 = 1./(1 + exp(-L2));
        NL2_hist = [NL2_hist, NL2];
        if NL2 < 0.5
            pred = 0;
        else
            pred = 1;
        end
        
        acc_local = [acc_local; y(i1), pred];

    %     y_star = (1-y(i1))/2;
        L = L - (y(i1) * log(NL2) + (1 - y(i1)) * log(1 - NL2));

        delta_i = y(i1) - NL2;
        dL_w2 = dL_w2 + delta_i*NL1_b.';
        dL_w2_hist = [dL_w2_hist, dL_w2];
        dL_w1 = dL_w1 + (NL1.*(1-NL1).* (W2(1:end-1)*(y(i1) - NL2))) * x_b(i1,:);
        dL_w1_hist = [dL_w1_hist, dL_w1];

    end
    accuracy = [accuracy; acc_local];
%     acc_p = [acc_p; sum(acc_local)/length(acc_local)];
    W2 = W2 + learning_rate .* dL_w2./length(x_b);
    W1 = W1 + learning_rate .* dL_w1./length(x_b);
    L_hist = [L_hist; L];
    acc_p = [acc_p, sum(accuracy(:,1) == accuracy(:,2))/length(accuracy)];
end
plot(acc_p, 'o')
% plot(L_hist, 'o')

end



