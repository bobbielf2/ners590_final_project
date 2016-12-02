function [A,An] = LapSLPmatrix(t,s,a) % single-layer kernel matrix & targ n-deriv
% t = target seg (x,nx cols), s = src seg, a = optional translation of src seg
% No jump included on self-interaction of derivative (ie PV integral).
if nargin==0, test_laplaceeval; return; end 
if nargin<3, a = 0; end
N = numel(s.x); M = numel(t.x);
d = repmat(t.x, [1 N]) - repmat(s.x.' + a, [M 1]);    % C-# displacements mat
A = -(1/2/pi) * log(abs(d)) .* repmat(s.w(:)', [M 1]);  % infty for self-int
if nargout==2                      % apply D^T
  nx = repmat(-t.nx, [1 N]);       % identical cols given by -targ normals
  An = (1/2/pi) * real(nx./d);     % complex form of dipole. Really A1 is An
  if numel(s.x)==numel(t.x) && max(abs(s.x+a-t.x))<1e-14
    An(diagind(An)) = -s.cur/4/pi; end  % self? diagonal term for Laplace
  An = An .* repmat(s.w(:)', [M 1]);
end
end

function i = diagind(A) % return indices of diagonal of square matrix
N = size(A,1); i = sub2ind(size(A), 1:N, 1:N);
end

function test_laplaceeval
close all
side = 'e'; % test interior or exterior
lptype = 's'; % test SLP or DLP
N = 1500;

% set up source and target
% source: starfish domain
a = .3; w = 5;           % smooth wobbly radial shape params...
R = @(t) (1 + a*cos(w*t))*1; Rp = @(t) -w*a*sin(w*t); Rpp = @(t) -w*w*a*cos(w*t);
s.Z = @(t) R(t).*exp(1i*t); s.Zp = @(t) (Rp(t) + 1i*R(t)).*exp(1i*t);
s.Zpp = @(t) (Rpp(t) + 2i*Rp(t) - R(t)).*exp(1i*t);
s = setupquad(s, N);

% target
nx = 100; gx = ((1:nx)/nx*2-1)*1.5; ny = 100; gy = ((1:ny)/ny*2-1)*1.5; % set up plotting grid
[xx, yy] = meshgrid(gx,gy); zz = (xx+1i*yy);
t = [];
[IN, ON] = inpolygon(real(zz),imag(zz),real(s.x),imag(s.x));
if side == 'i'
    ii = IN & ~ON;
elseif side == 'e'
    ii = ~IN;
end
t.x = zz(ii(:));  % eval pts only on one side

% generate the exact solution

uexa = nan*(1+1i)*zz; % exact soln
A = LapSLPmatrix(t,s,0);
tau = sin(abs(s.x))+cos(abs(s.x));
u_temp = A*tau;
uexa(ii(:)) = u_temp; % the exact soln (complex form)

% plot the exact product
uexa(IN) = 0;
figure()
surf(xx,yy,uexa)

figure()
plot(real(s.x),imag(s.x))

% SLP test
Nn = 5;
err = NaN(Nn,1);
for NN = 1:Nn
    N = 100*NN;
    s = setupquad(s, N);
     
    u = nan*(1+1i)*zz;
    A = LapSLPmatrix(t,s,0);
    tau = sin(abs(s.x))+cos(abs(s.x));
    u_temp = A*tau;
    u(ii(:)) = u_temp; 

    err(NN) = max(abs(u(:)-uexa(:)));
end

figure(),clf, imagesc(gx,gy,log10(abs(u-uexa))), colorbar, title('log10 err in |u|'), axis equal tight
figure(),clf, semilogy(100*(1:Nn),err,'o'), title('BVP conv')
end
