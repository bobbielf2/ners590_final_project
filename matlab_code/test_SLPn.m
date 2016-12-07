function test_SLPn
close all
side = 'e'; % test interior or exterior
lptype = 's'; % test SLP or DLP
x = generateCircles();
[N,M] = size(x);    % n #nodes per particle, m number of particle


%% set up source
for l = 1:M
    s = [];
    s.x = x(:,l);
    s = setupquad(s, N);
    eval(['s_' num2str(l) ' = s;'])
end


%% set up target
nx = 100; gx = (1:nx)/nx; ny = 100; gy = (1:ny)/ny; % set up plotting grid
[xx, yy] = meshgrid(gx,gy); zz = (xx+1i*yy);
t = [];
ii = ones(size(xx));
for l = 1:M
    eval(['s = s_' num2str(l)])
    [IN, ON] = inpolygon(real(zz),imag(zz),real(s.x),imag(s.x));
    if side == 'i'
        ii = IN & ~ON & ii;
    elseif side == 'e'
        ii = ~IN & ii;
    end
end
 
figure()
imagesc(gx,gy,ii)    % test whether inpolygon is working correctly
caxis([-0.25,1.25])
colorbar
title('target point location') 
axis equal tight
t.x = zz(ii(:));  

%% multipole evaluation
u = 0*(1+1i)*zz; 
for l = 1:M
    eval(['s = s_' num2str(l)])
    A = LapSLPmatrix(t,s,0);
    tau = sin(2*pi*real(s.x))+cos(pi*imag(s.x));
    u_temp = A*tau;
    u(ii(:)) = u(ii(:)) + u_temp;
    if mod(l,25) == 1
        figure()
        imagesc(gx,gy,u) 
        %caxis([0, 4]) 
        colorbar
        title(sprintf('Result after %d particle', l)) 
        axis equal tight
    end
end
figure() 
imagesc(gx,gy,u)
%caxis([0, 4])
colorbar
title(sprintf('Result after %d particle', l))
axis equal tight
