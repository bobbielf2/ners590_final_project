% Test Lap close evaluation
% only DLP now, SLP to do
function test_LapCloseEval()
n = 200; % #nodes per particle
side = 'i'; % interior or exterior

%% setup geometry
s.Z = @(t) (1 + 0.3 * cos(5 * t)) .* exp(1j * t);
R = @(t) 1 + 0.3 * cos(5 * t);
s.inside = @(z) abs(z) < R(angle(z));
s.outside = @(z) abs(z) > R(angle(z));
s = setupquad(s, n);

%% build exact solution
a = 1.1+1j;
if side == 'e'
    a = 0.1+0.3j;
end
f = @(z) 1./(z-a);
fp = @(z) -1./(z-a).^2;

% solution grid
ub = real(f(s.x));
nx = 150;
gx = ((1:nx)/nx*2-1)*1.5;
gy = gx;
[xx, yy] = meshgrid(gx,gy);
zz = xx + 1j*yy;

if side == 'e'
    ii = s.outside(zz);
else
    ii = s.inside(zz);
end
t.x = zz(ii);

uexa = nan(size(zz));
uexa(ii) = real(f(t.x));
subplot(2,2,1)
imagesc(gx,gy,uexa)
colorbar
title('exact')

%% solve for density tau
if side=='i'
    A = -eye(n)/2 + LapDLPmatrix(s,s);  % full rank
else
    A = eye(n)/2 + LapDLPmatrix(s,s);   % has rank-1 nullspace, ok for dense solve
end
tau = A\ub;

%% close eval
u = nan(size(zz));
% up = nan(size(zz));
u(ii) = lapDevalclose(t.x,s,tau,side);
subplot(2,2,3)
imagesc(gx,gy,u)
colorbar
title('close eval')

% error
subplot(2,2,4)
imagesc(gx,gy,log10(abs(u-uexa)))
colorbar
title('error')