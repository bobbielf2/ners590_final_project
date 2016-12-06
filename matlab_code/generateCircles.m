function x = generateCircles()
% x = GENERATECIRCLES(v)
% 
% generate random heterogeneous circles

v = 1;                  % verbosity: 1=plot circles, 0=don't plot
m = 100;                % number of particles to generate
n = 64;                 % #nodes per particle
offset = 0.01;          % smallest distance between particles
rmax = sqrt(1/m/pi);    % max radius
rmin = 1/2/m;           % min radius

x = zeros(n,m);         % node coordinates of particles
c = zeros(m,2);         % centers of particles
r = zeros(m,1);         % radii of particles
t = linspace(0,2*pi,n+1).'; t(end) = [];

j = 0;
while j<m
    j = j+1;
    flag = false;
    while ~flag
        r(j) = rmin + rand()*(rmax-rmin);
        c(j,:) = r(j)+offset + rand(1,2)*(1-2*r(j)-2*offset);
        flag = true;
        if any(abs((c(1:j-1,1)-c(j,1)) + 1i*(c(1:j-1,2)-c(j,2))) < r(1:j-1) + r(j) + offset)
            flag = false;
        end
    end
    x(:,j) = r(j)*exp(1i*t)+c(j,1)+1i*c(j,2);
end

if v
    figure(1)
    clf
    for j = 1:m
        plot(x([1:end,1],j),'.')
        hold on
    end
    axis([0,1,0,1])
    axis square
end