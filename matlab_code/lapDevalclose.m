function [u ux uy do imv] = lapDevalclose(x, s, tau, side) 
% LAPDEVALCLOSE - Laplace potential and field for DLP on curve (global quadr)
%
% u = lapDevalclose(x,s,tau,side) returns potentials at targets x due to DLP
%  with real-valued density tau living on curve s with global periodic
%  trapezoid rule. "side" controls whether targets are inside (default) or
%  outside. A scheme based on Helsing/Ioakimidis globally compensated quadr is
%  used, except for the exterior when a new scheme is used.
%
% Our definition of the DLP on curve \Gamma is, associating R^2 with the complex
% plane, with x = (x_1,x_2) a point in the complex plane,
%
%    u(x) = Re (1/2\pi i) \int_\Gamma \tau(y) / (x-y) dy
%
% [u ux uy] = lapDevalclose(x,s,tau,side) also returns field (the two first
%  partial derivatives of potential u)
%
% inputs:
% x = M-by-1 list of targets, as points in complex plane
% s = curve struct containing N-by-1 vector s.x of source nodes (as complex
%     numbers), and all other fields in s generated by quadr(), and
%     s.a one interior point far from bdry (mean(s.x) used if not provided)
% tau = double-layer density values at nodes. Note, must be real-valued.
% side = 'i','e' to indicate targets are interior or exterior to the curve.
%
% outputs:
% u     = potential values at targets x (M-by-1)
% ux,uy = (optional) first partials of potential at targets
% do    = (optional) diagnostic outputs:
%   do.vb : complex values on bdry (ie, v^+ for side='e' or v^- for side='i')
% imv   = imag part of v at targets (M-by-1)
%
% complexity O(N^2) for evaluation of v^+ or v^-, plus O(NM) for globally
% compensated quadrature to targets
%
% (C) Barnett 10/9/13. Switched to call cauchycompeval 10/23/13

if nargin<4, side='i'; end
wantder = nargout>1;
if ~isfield(s,'a'), s.a = mean(s.x);
  %warning('I''m guessing s.a for you. I sure hope it''s inside the curve!');
end

% Helsing step 1: eval bdry limits at nodes of v = complex DLP(tau)...
vb = 0*s.x;                % will become v^+ or v^- bdry data of holom func
taup = perispecdiff(tau);  % parameter-deriv of tau
N = numel(s.x);
for i=1:N, j = [1:i-1, i+1:N];   % skip pt i
  vb(i) = sum((tau(j)-tau(i))./(s.x(j)-s.x(i)).*s.cw(j)) + taup(i)*s.w(i)/s.sp(i);
end
vb = vb*(1/(-2i*pi));   % prefactor
if side=='i', vb = vb - tau; end % JR's add for v^-, cancel for v^+
do.vb = vb;

% Helsing step 2: compensated close-evaluation of u = Re(v) & its deriv...
if wantder, [v vp] = cauchycompeval(x,s,vb,side);
  ux = real(vp); uy = -imag(vp);
else v = cauchycompeval(x,s,vb,side); end
u = real(v);
imv = imag(v);

function fp = perispecdiff(f) % spectral differentiation sampled periodic func
N = numel(f);          % can be row or col vec
fh = fft(f);
if mod(N,2)==0, fp = ifft(1i*[0:N/2-1, 0, -N/2+1:-1]' .* fh(:));  % even case
else, fp = ifft(1i*[0:(N-1)/2, -(N-1/2):-1]' .* fh(:)); end       % odd
fp = reshape(fp, size(f));

function testperispecdiff
N = 50; tj = 2*pi/N*(1:N)';
f = sin(3*tj); fp = 3*cos(3*tj);   % trial periodic function & its deriv
norm(fp-perispecdiff(f))
