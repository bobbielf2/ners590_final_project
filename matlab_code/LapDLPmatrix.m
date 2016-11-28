function [A, An] = LapDLPmatrix(t,s,a) % double-layer kernel matrix & targ n-deriv
% t = target seg (x,nx cols), s = src seg, a = optional translation of src seg
% No jump included on self-interaction (ie principal-value integral)
if nargin<3, a = 0; end
N = numel(s.x); M = numel(t.x);
d = repmat(t.x, [1 N]) - repmat(s.x.' + a, [M 1]);    % C-# displacements mat
ny = repmat(s.nx.', [M 1]);      % identical rows given by src normals
A = (1/2/pi) * real(ny./d);      % complex form of dipole
if numel(s.x)==numel(t.x) && max(abs(s.x+a-t.x))<1e-14
  A(diagind(A)) = -s.cur/4/pi; end           % self? diagonal term for Laplace
A = A .* repmat(s.w(:)', [M 1]);
if nargout>1 % deriv of double-layer. Not correct for self-interaction.
  csry = conj(ny).*d;              % (cos phi + i sin phi).r
  nx = repmat(t.nx, [1 N]);        % identical cols given by target normals
  csrx = conj(nx).*d;              % (cos th + i sin th).r
  r = abs(d);    % dist matrix R^{MxN}
  An = -real(csry.*csrx)./(r.^2.^2)/(2*pi);
  An = An .* repmat(s.w(:)', [M 1]);
end
end

function i = diagind(A) % return indices of diagonal of square matrix
N = size(A,1); i = sub2ind(size(A), 1:N, 1:N);
end