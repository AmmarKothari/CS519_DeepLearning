close all
clear all
n = 3;
iter = 10000;
holder = zeros(iter, n);
min_dist = zeros(iter,1);
for i1 = 1:iter
    holder(i1, :) = rand(1,n);
    min_dist(i1) = min(holder(i1,:));
end

disp(median(min_dist))
% plot(med)



%%
points  = 1000;
n = 100;
var = zeros(n, 1);
for i = 1:n
    var = var + rand(1, points);
end
var = var ./ n;
% histogram(var1)
histogram(var)